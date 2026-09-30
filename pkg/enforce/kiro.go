package enforce

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/obot-platform/obot-sentry/pkg/localagent"
	"github.com/obot-platform/obot-sentry/pkg/toolkind"
	"github.com/obot-platform/obot/apiclient/types"
	"gopkg.in/yaml.v3"
)

// Kiro's PreToolUse payload, as the Kiro 1.2 agent extension builds it:
// session_id, hook_event_name, cwd (the first workspace root), tool_name (the
// tool's id), and tool_input. It names no MCP server, so every MCP call is
// resolved from the tool id and Kiro's own configuration files.
type kiroPreToolPayload struct {
	ToolName  string          `json:"tool_name"`
	ToolInput json.RawMessage `json:"tool_input,omitempty"`
	SessionID string          `json:"session_id,omitempty"`
	CWD       string          `json:"cwd,omitempty"`
}

// Kiro tool ids that carry their MCP target in tool_input rather than in the id.
const (
	// kiroPowersTool runs a Power's MCP tool when action is "use", with the
	// power, server, and tool named in the input. Its other actions (activate,
	// readSteering, readSkill) read the Power's own files.
	kiroPowersTool = "kiro_powers"
	// kiroToolCall runs a deferred MCP tool by "<server>::<tool>" id. Kiro fires
	// hooks for the tool it runs rather than for this wrapper, so a hook should
	// never see it; it is handled so that one that does can't pass as built-in.
	kiroToolCall = "tool_call"
)

// kiroToolIDMax is Kiro's cap on an MCP tool id; longer ids are truncated.
const kiroToolIDMax = 64

var kiroIDUnsafe = regexp.MustCompile(`[^a-zA-Z0-9_]`)

// kiroSanitize is the transform Kiro applies to "<server>_<tool>" when it builds
// an MCP tool id (v3u in the extension): whitespace and hyphens become
// underscores, everything else outside [a-zA-Z0-9_] is dropped, and the result
// is lowercased. It is applied character by character, so it distributes over
// the join and a server's contribution can be computed on its own.
func kiroSanitize(s string) string {
	s = strings.NewReplacer(" ", "_", "\t", "_", "\n", "_", "\r", "_", "-", "_").Replace(s)
	return strings.ToLower(kiroIDUnsafe.ReplaceAllString(s, ""))
}

// kiroServerPrefix is the part of a tool id contributed by a server key.
func kiroServerPrefix(key string) string {
	return "mcp_" + kiroSanitize(key) + "_"
}

func normalizeKiroPreTool(ctx context.Context, env Env, raw []byte) (Call, error) {
	var hook kiroPreToolPayload
	if err := decodePayload(raw, &hook); err != nil {
		return Call{}, err
	}
	toolName := strings.TrimSpace(hook.ToolName)
	if toolName == "" {
		return Call{}, errors.New("pre-tool hook payload has no tool_name")
	}
	if err := boundedField("tool_name", toolName, maxToolNameBytes); err != nil {
		return Call{}, err
	}
	if err := boundedField("cwd", hook.CWD, maxWorkingDirBytes); err != nil {
		return Call{}, err
	}

	call := Call{Request: types.EnforcementDecisionRequest{Agent: wireAgentKiro, Tool: toolName}}
	target, ok := kiroMCPTarget(toolName, hook.ToolInput)
	if !ok {
		call.Request.Kind = toolkind.KiroKind(toolName)
		return call, nil
	}
	call.Request.Kind = toolkind.KindMCP

	loader := newConfigLoader()
	tr := &tracer{}
	res, tool := resolveKiro(ctx, loader, env, hook.CWD, target, tr)
	res.Trace = tr.steps
	call.Trace = res.Trace
	call.Request.Tool = tool
	call.Request.ServerName = res.ServerName
	call.Request.Server = res.Identity
	call.Request.Unresolved = res.Unresolved
	if res.Unresolved {
		call.Request.UnresolvedReason = res.Reason
	}
	return call, nil
}

// kiroTarget is what a Kiro tool call says about its MCP server before any
// configuration is read: either an exact server key and tool (kiro_powers,
// tool_call), or a lossy tool id to match against configured keys.
type kiroTarget struct {
	// key is the exact configuration key, when the call names one.
	key string
	// tool is the tool within the server, when the call names it exactly.
	tool string
	// id is the sanitized tool id, when the call only carries that.
	id string
	// invalid is a reason the call is MCP but names nothing usable.
	invalid string
}

// kiroMCPTarget reports whether a Kiro tool call is an MCP call, and what it
// says about its target.
func kiroMCPTarget(toolName string, input json.RawMessage) (kiroTarget, bool) {
	switch toolName {
	case kiroPowersTool:
		var in struct {
			Action     string `json:"action"`
			PowerName  string `json:"powerName"`
			ServerName string `json:"serverName"`
			ToolName   string `json:"toolName"`
		}
		_ = json.Unmarshal(input, &in)
		if in.Action != "use" {
			return kiroTarget{}, false
		}
		power, server := strings.TrimSpace(in.PowerName), strings.TrimSpace(in.ServerName)
		if power == "" || server == "" {
			return kiroTarget{invalid: "the Kiro Powers call did not name its power and MCP server"}, true
		}
		return kiroTarget{key: kiroPowerServerKey(power, server), tool: strings.TrimSpace(in.ToolName)}, true
	case kiroToolCall:
		var in struct {
			ToolID string `json:"tool_id"`
		}
		_ = json.Unmarshal(input, &in)
		server, tool, ok := strings.Cut(strings.TrimSpace(in.ToolID), "::")
		if !ok || server == "" {
			return kiroTarget{invalid: "the deferred tool call did not name its MCP server"}, true
		}
		return kiroTarget{key: server, tool: tool}, true
	}
	if toolkind.KiroKind(toolName) != toolkind.KindMCP {
		return kiroTarget{}, false
	}
	return kiroTarget{id: toolName}, true
}

// kiroPowerServerKey is the name Kiro registers a Power's server under (Qne in
// the extension), both in the settings file's powers section and for an Agent
// Plugins power's own mcp.json.
func kiroPowerServerKey(power, server string) string {
	return "power-" + power + "-" + server
}

// resolveKiro resolves a Kiro MCP call and returns the tool name to report.
func resolveKiro(ctx context.Context, loader *configLoader, env Env, cwd string, target kiroTarget, tr *tracer) (Resolution, string) {
	if target.invalid != "" {
		return unresolved("", target.invalid), ""
	}
	if target.key != "" {
		return resolveKiroKey(ctx, loader, env, cwd, target.key, tr), target.tool
	}

	// The id is mcp_<sanitized server>_<sanitized tool>, so it names no server by
	// itself: a server is a candidate when its key's own contribution is a prefix
	// of the id. More than one candidate is two readings of one name, and like an
	// mcp__ name that splits two ways (resolveSplits), picking one could report
	// an allowlisted server for a call that went to another.
	keys := kiroCandidateKeys(ctx, loader, env, cwd, target.id)
	switch len(keys) {
	case 0:
		kiroTraceAll(ctx, loader, env, cwd, tr)
		return unresolved("", fmt.Sprintf(
			"no Kiro MCP configuration declares a server whose tools are named %q", target.id)), target.id
	case 1:
		return resolveKiroKey(ctx, loader, env, cwd, keys[0], tr), kiroToolOf(target.id, keys[0])
	default:
		distinct := map[string]bool{}
		for _, k := range keys {
			distinct[kiroServerPrefix(k)] = true
		}
		if len(distinct) == 1 {
			// Keys that sanitize alike ("my-server", "my_server") produce the same
			// ids; they are one reading only if they define the same server.
			if res, ok := kiroAgreeingKeys(ctx, loader, env, cwd, keys, tr); ok {
				return res, kiroToolOf(target.id, keys[0])
			}
		}
		return ambiguousToolName(localagent.Kiro, target.id, keys), target.id
	}
}

// kiroToolOf is the tool half of id once key's prefix is removed. It is Kiro's
// sanitized, lowercased form of the tool name, possibly truncated or suffixed
// with _<n> where two tools collided: the real name is not recoverable offline.
func kiroToolOf(id, key string) string {
	if rest, ok := strings.CutPrefix(id, kiroServerPrefix(key)); ok {
		return rest
	}
	return id
}

// kiroIDMatches reports whether a tool id can have come from server key.
func kiroIDMatches(id, key string) bool {
	prefix := kiroServerPrefix(key)
	if strings.HasPrefix(id, prefix) && len(id) > len(prefix) {
		return true
	}
	// A key long enough that the id cap cut into its own prefix.
	return len(id) == kiroToolIDMax && strings.HasPrefix(prefix, id)
}

// kiroCandidateKeys returns every configured server key, across all of Kiro's
// scopes, that the tool id can have come from.
func kiroCandidateKeys(ctx context.Context, loader *configLoader, env Env, cwd, id string) []string {
	seen := map[string]bool{}
	for _, s := range append(kiroScopes(loader, env, cwd), kiroAgentScopes(loader, env, cwd)...) {
		set, res := s.load(ctx)
		if res != loadOK {
			continue
		}
		for key := range set {
			if kiroIDMatches(id, key) {
				seen[key] = true
			}
		}
	}
	keys := make([]string, 0, len(seen))
	for key := range seen {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}

// kiroAgreeingKeys resolves several keys and succeeds only when they all
// identify the same server.
func kiroAgreeingKeys(ctx context.Context, loader *configLoader, env Env, cwd string, keys []string, tr *tracer) (Resolution, bool) {
	var first Resolution
	for i, key := range keys {
		res := resolveKiroKey(ctx, loader, env, cwd, key, tr)
		if res.Unresolved {
			return Resolution{}, false
		}
		if i == 0 {
			first = res
			continue
		}
		if res.Identity != first.Identity {
			return Resolution{}, false
		}
	}
	return first, true
}

// resolveKiroKey resolves an exact server key.
//
// Kiro merges MCP configuration with the workspace over the user file, and a
// custom agent's own mcpServers over both, but the payload does not say which
// agent is running. So agent profiles are peers of whatever the files resolve
// to: a name an agent profile defines differently from the workspace or user
// file is ambiguous. Letting the profile win would let an inactive profile's
// allowlisted definition answer for the server that actually ran, and every
// one of these files is the user's to edit.
func resolveKiroKey(ctx context.Context, loader *configLoader, env Env, cwd, key string, tr *tracer) Resolution {
	names := lookup{names: []string{key}}
	fileMatch, fileOut := resolveScopes(ctx, kiroScopes(loader, env, cwd), names, tr)
	agentMatch, agentOut := resolveScopes(ctx, kiroAgentScopes(loader, env, cwd), names, tr)
	if fileOut == outcomeAmbiguous || agentOut == outcomeAmbiguous {
		return ambiguous(localagent.Kiro, key)
	}
	switch {
	case fileOut == outcomeFound && agentOut == outcomeFound:
		if !sameEntry(fileMatch.entry, agentMatch.entry) {
			return ambiguous(localagent.Kiro, key)
		}
		return resolved(env, fileMatch.key, fileMatch.entry)
	case fileOut == outcomeFound:
		return resolved(env, fileMatch.key, fileMatch.entry)
	case agentOut == outcomeFound:
		return resolved(env, agentMatch.key, agentMatch.entry)
	default:
		return notFound(localagent.Kiro, key, fmt.Sprintf(
			"MCP server %q was not found in any Kiro MCP configuration", key))
	}
}

// kiroTraceAll records every scope as consulted, for a tool id no key matched.
func kiroTraceAll(ctx context.Context, loader *configLoader, env Env, cwd string, tr *tracer) {
	for _, s := range append(kiroScopes(loader, env, cwd), kiroAgentScopes(loader, env, cwd)...) {
		_, res := s.load(ctx)
		tr.miss(s.path, s.traceKey(""), res)
	}
}

// kiroScopes returns Kiro's MCP configuration files in precedence order:
//
//  0. the workspace's .kiro/settings/mcp.json, from the payload's cwd;
//  1. the user's ~/.kiro/settings/mcp.json;
//  2. the servers Powers contribute, as peers: the user file's powers section,
//     where legacy (POWER.md) powers are registered, and each installed Agent
//     Plugins power's own mcp.json. Both name their servers power-<power>-<server>.
//
// Kiro merges the mcp.json of every open workspace root, but the payload carries
// only the first as cwd, so a server declared only by another root does not
// resolve and the call is denied.
func kiroScopes(loader *configLoader, env Env, cwd string) []scope {
	var scopes []scope
	if cwd = strings.TrimSpace(cwd); filepath.IsAbs(cwd) {
		path := filepath.Join(filepath.Clean(cwd), ".kiro", "settings", "mcp.json")
		scopes = append(scopes, scope{path: path, key: mcpServersKey, rank: 0, load: jsonServers(loader, path)})
	}
	user := env.homePath(".kiro", "settings", "mcp.json")
	scopes = append(scopes,
		scope{path: user, key: mcpServersKey, rank: 1, load: jsonServers(loader, user)},
		scope{path: user, key: "powers.mcpServers", rank: 2, load: kiroPowersSectionServers(loader, user)},
	)
	for _, power := range kiroInstalledPowers(loader, env) {
		path := env.homePath(".kiro", "powers", "installed", power, "mcp.json")
		scopes = append(scopes, scope{path: path, key: mcpServersKey, rank: 2, load: kiroPluginPowerServers(loader, path, power)})
	}
	return scopes
}

// kiroPowersSectionServers loads the settings file's powers.mcpServers table.
func kiroPowersSectionServers(loader *configLoader, path string) func(context.Context) (serverSet, loadResult) {
	return func(ctx context.Context) (serverSet, loadResult) {
		var doc struct {
			Powers struct {
				MCPServers map[string]json.RawMessage `json:"mcpServers"`
			} `json:"powers"`
		}
		if res := loader.loadJSON(ctx, path, &doc); res != loadOK {
			return nil, res
		}
		return decodeServers(doc.Powers.MCPServers), loadOK
	}
}

// kiroPluginPowerServers loads an Agent Plugins power's mcp.json, keyed the way
// Kiro registers its servers.
func kiroPluginPowerServers(loader *configLoader, path, power string) func(context.Context) (serverSet, loadResult) {
	return func(ctx context.Context) (serverSet, loadResult) {
		set, res := jsonServers(loader, path)(ctx)
		if res != loadOK {
			return nil, res
		}
		out := make(serverSet, len(set))
		for name, entry := range set {
			out[kiroPowerServerKey(power, name)] = entry
		}
		return out, loadOK
	}
}

// kiroInstalledPowers lists the Agent Plugins powers Kiro loads: those named in
// ~/.kiro/powers/installed.json whose directory has a plugin.json. A legacy
// power's servers are already in the settings file's powers section.
func kiroInstalledPowers(loader *configLoader, env Env) []string {
	var doc struct {
		InstalledPowers []struct {
			Name string `json:"name"`
		} `json:"installedPowers"`
	}
	if loader.loadJSON(context.Background(), env.homePath(".kiro", "powers", "installed.json"), &doc) != loadOK {
		return nil
	}
	var out []string
	for _, p := range doc.InstalledPowers {
		name := p.Name
		// Kiro's own guard: one plain path segment.
		if name == "" || name == "." || name == ".." || strings.ContainsAny(name, `/\`+"\x00") {
			continue
		}
		if _, err := os.Stat(env.homePath(".kiro", "powers", "installed", name, "plugin.json")); err == nil {
			out = append(out, name)
		}
	}
	return out
}

// kiroMaxAgentFiles bounds how many custom agent profiles a hook reads, so a
// huge agents tree can't stall a tool call past the hook's budget.
const kiroMaxAgentFiles = 256

// kiroAgentScopes returns every custom agent profile's own mcpServers table,
// all as peers: the user's ~/.kiro/agents and the workspace's .kiro/agents,
// .json profiles and .md profiles with YAML frontmatter, searched recursively
// the way Kiro loads them.
func kiroAgentScopes(loader *configLoader, env Env, cwd string) []scope {
	dirs := []string{env.homePath(".kiro", "agents")}
	if cwd = strings.TrimSpace(cwd); filepath.IsAbs(cwd) {
		dirs = append(dirs, filepath.Join(filepath.Clean(cwd), ".kiro", "agents"))
	}
	var scopes []scope
	for _, dir := range dirs {
		_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
			if err != nil || len(scopes) >= kiroMaxAgentFiles {
				if d != nil && d.IsDir() {
					return fs.SkipDir
				}
				return nil
			}
			if d.IsDir() {
				return nil
			}
			switch filepath.Ext(path) {
			case ".json":
				scopes = append(scopes, scope{path: path, key: mcpServersKey, load: jsonServers(loader, path)})
			case ".md":
				scopes = append(scopes, scope{path: path, key: mcpServersKey, load: kiroMarkdownAgentServers(loader, path)})
			}
			return nil
		})
	}
	return scopes
}

// kiroMarkdownAgentServers loads the mcpServers table from a Markdown agent
// profile's YAML frontmatter.
func kiroMarkdownAgentServers(loader *configLoader, path string) func(context.Context) (serverSet, loadResult) {
	return func(ctx context.Context) (serverSet, loadResult) {
		data, res := loader.readConfig(ctx, path)
		if res != loadOK {
			return nil, res
		}
		text := bytes.TrimLeft(data, " \t\r\n")
		if !bytes.HasPrefix(text, []byte("---")) {
			return serverSet{}, loadOK
		}
		block, _, found := bytes.Cut(bytes.TrimLeft(text[3:], "\r\n"), []byte("\n---"))
		if !found {
			return serverSet{}, loadOK
		}
		var fm struct {
			MCPServers map[string]any `yaml:"mcpServers"`
		}
		if yaml.Unmarshal(block, &fm) != nil {
			return nil, loadUnusable
		}
		raw := make(map[string]json.RawMessage, len(fm.MCPServers))
		for name, v := range fm.MCPServers {
			if b, err := json.Marshal(v); err == nil {
				raw[name] = b
			}
		}
		return decodeServers(raw), loadOK
	}
}
