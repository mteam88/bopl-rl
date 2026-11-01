using System.Diagnostics;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Operations;

// Analyze the immutable source pin, not the growing reference instrumentation.
// This is conservative type reachability, not a proof of runtime reachability.
if (args.Length != 2)
    throw new ArgumentException("source-audit <bopl-rl checkout root (with the game/ submodule)> <output.json>");
var root = Path.GetFullPath(args[0]);
using var pin = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, "native/data/full-reference.json")));
var revision = pin.RootElement.GetProperty("commit").GetString()!;
var sourceHashes = pin.RootElement.GetProperty("files");
var serializedPaths = sourceHashes.EnumerateObject().Select(x => x.Name)
    .Where(x => x.EndsWith(".prefab", StringComparison.Ordinal) || x.EndsWith(".unity", StringComparison.Ordinal)
        || x.EndsWith(".asset", StringComparison.Ordinal)).OrderBy(x => x, StringComparer.Ordinal).ToArray();
var start = new ProcessStartInfo("git") {
    WorkingDirectory = Path.Combine(root, "game"), RedirectStandardOutput = true, RedirectStandardError = true
};
foreach (var arg in new[] { "archive", "--format=zip", revision, "Assets/Scripts" })
    start.ArgumentList.Add(arg);
foreach (var path in serializedPaths) start.ArgumentList.Add(path);
using var process = Process.Start(start)!;
var errors = process.StandardError.ReadToEndAsync();
using var archiveBytes = new MemoryStream();
await process.StandardOutput.BaseStream.CopyToAsync(archiveBytes);
await process.WaitForExitAsync();
if (process.ExitCode != 0) throw new Exception(await errors);
archiveBytes.Position = 0;
using var archive = new ZipArchive(archiveBytes);
var trees = new List<SyntaxTree>();
var files = new SortedDictionary<string, string>(StringComparer.Ordinal);
foreach (var entry in archive.Entries.Where(x => x.FullName.EndsWith(".cs", StringComparison.Ordinal))
             .OrderBy(x => x.FullName, StringComparer.Ordinal)) {
    using var stream = entry.Open();
    using var bytes = new MemoryStream();
    stream.CopyTo(bytes);
    var content = bytes.ToArray();
    var hash = Convert.ToHexString(SHA256.HashData(content)).ToLowerInvariant();
    if (!sourceHashes.TryGetProperty(entry.FullName, out var expected) || expected.GetString() != hash)
        throw new Exception("Source pin mismatch: " + entry.FullName);
    files.Add(entry.FullName, hash);
    var text = Encoding.UTF8.GetString(content).TrimStart('\uFEFF');
    trees.Add(CSharpSyntaxTree.ParseText(text, new CSharpParseOptions(LanguageVersion.CSharp9), entry.FullName));
}
var managed = Path.Combine(root, "game/Builds/Linux/BoplBattle_Data/Managed");
var assemblies = Directory.GetFiles(managed, "*.dll")
    .Where(x => Path.GetFileName(x) != "Assembly-CSharp.dll")
    .OrderBy(x => x, StringComparer.Ordinal).ToArray();
var compilation = CSharpCompilation.Create("Assembly-CSharp", trees,
    assemblies.Select(x => MetadataReference.CreateFromFile(x)),
    new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true, checkOverflow: false));
var diagnostics = compilation.GetDiagnostics().Where(x => x.Severity == DiagnosticSeverity.Error)
    .Select(x => x.ToString()).OrderBy(x => x, StringComparer.Ordinal).ToArray();
if (diagnostics.Length != 0) {
    Console.Error.WriteLine(string.Join("\n", diagnostics.Take(30)));
    throw new Exception($"Pinned source has {diagnostics.Length} compiler errors; refusing incomplete semantic audit.");
}

string TypeId(INamedTypeSymbol symbol) => symbol.OriginalDefinition.ToDisplayString();
var types = new SortedDictionary<string, TypeData>(StringComparer.Ordinal);
foreach (var tree in trees) {
    var model = compilation.GetSemanticModel(tree);
    foreach (var node in tree.GetRoot().DescendantNodes().Where(x => x is BaseTypeDeclarationSyntax or DelegateDeclarationSyntax)) {
        if (model.GetDeclaredSymbol(node) is not INamedTypeSymbol symbol) continue;
        var id = TypeId(symbol);
        if (!types.TryGetValue(id, out var data)) {
            data = new TypeData { Name = id, Kind = symbol.TypeKind.ToString(),
                Base = symbol.BaseType?.ToDisplayString(),
                Interfaces = symbol.Interfaces.Select(TypeId).OrderBy(x => x, StringComparer.Ordinal).ToArray() };
            types.Add(id, data);
        }
        var span = node.GetLocation().GetLineSpan();
        data.Declarations.Add(new Site(tree.FilePath, span.StartLinePosition.Line + 1));
        data.Lines += span.EndLinePosition.Line - span.StartLinePosition.Line + 1;
    }
}
var featureKinds = new HashSet<SyntaxKind> {
    SyntaxKind.AnonymousMethodExpression, SyntaxKind.SimpleLambdaExpression, SyntaxKind.ParenthesizedLambdaExpression,
    SyntaxKind.AwaitExpression, SyntaxKind.YieldReturnStatement, SyntaxKind.YieldBreakStatement,
    SyntaxKind.UnsafeStatement, SyntaxKind.FixedStatement, SyntaxKind.StackAllocArrayCreationExpression,
    SyntaxKind.TryStatement, SyntaxKind.ThrowStatement, SyntaxKind.ThrowExpression,
    SyntaxKind.LockStatement, SyntaxKind.QueryExpression, SyntaxKind.TypeOfExpression
};
foreach (var tree in trees) {
    var model = compilation.GetSemanticModel(tree);
    foreach (var node in tree.GetRoot().DescendantNodes()) {
        var declaration = node.Ancestors().FirstOrDefault(x => x is BaseTypeDeclarationSyntax or DelegateDeclarationSyntax);
        if (declaration == null || model.GetDeclaredSymbol(declaration) is not INamedTypeSymbol owner) continue;
        var data = types[TypeId(owner)];
        if (featureKinds.Contains(node.Kind())) {
            var kind = node.Kind().ToString();
            data.Features[kind] = data.Features.GetValueOrDefault(kind) + 1;
        }
        if (node is not SimpleNameSyntax && node is not PredefinedTypeSyntax) continue;
        var symbol = model.GetSymbolInfo(node).Symbol;
        if (symbol is IAliasSymbol alias) symbol = alias.Target;
        if (symbol == null || symbol is INamespaceSymbol or ILocalSymbol or IParameterSymbol or ITypeParameterSymbol) continue;
        var target = symbol as INamedTypeSymbol ?? symbol.ContainingType;
        if (target == null) continue;
        var targetId = TypeId(target);
        if (types.ContainsKey(targetId)) {
            if (targetId != data.Name) data.SourceTypes.Add(targetId);
        } else {
            var member = symbol.OriginalDefinition.ToDisplayString();
            var key = target.ContainingAssembly.Identity.Name + ":" + member;
            if (!data.External.TryGetValue(key, out var external)) {
                external = new ExternalData { Assembly = target.ContainingAssembly.Identity.Name,
                    Type = targetId, Member = member, Kind = symbol.Kind.ToString() };
                data.External.Add(key, external);
            }
            external.References++;
            if (external.Examples.Count < 3)
                external.Examples.Add(new Site(tree.FilePath, node.GetLocation().GetLineSpan().StartLinePosition.Line + 1));
        }
    }
}

// Overloaded Unity null/boolean tests and conversions have no explicit member
// name in C#. Include their bound operator symbols separately from name syntax.
foreach (var tree in trees) {
    var model = compilation.GetSemanticModel(tree);
    foreach (var node in tree.GetRoot().DescendantNodes()) {
        var operationRoot = node switch {
            BaseMethodDeclarationSyntax or AccessorDeclarationSyntax => node,
            PropertyDeclarationSyntax { ExpressionBody: not null } p => p.ExpressionBody.Expression,
            IndexerDeclarationSyntax { ExpressionBody: not null } p => p.ExpressionBody.Expression,
            EqualsValueClauseSyntax { Parent: VariableDeclaratorSyntax { Parent.Parent: FieldDeclarationSyntax } } => node,
            EqualsValueClauseSyntax { Parent: PropertyDeclarationSyntax } => node,
            _ => null
        };
        if (operationRoot == null) continue;
        var declaration = node.Ancestors().OfType<BaseTypeDeclarationSyntax>().FirstOrDefault();
        if (declaration == null || model.GetDeclaredSymbol(declaration) is not INamedTypeSymbol owner) continue;
        var data = types[TypeId(owner)];
        new OperatorVisitor((method, syntax) => {
            var targetId = TypeId(method.ContainingType);
            if (types.ContainsKey(targetId)) {
                if (targetId != data.Name) data.SourceTypes.Add(targetId);
                return;
            }
            var member = method.OriginalDefinition.ToDisplayString();
            var assembly = method.ContainingAssembly.Identity.Name;
            var key = assembly + ":" + member;
            if (!data.Operators.TryGetValue(key, out var external)) {
                external = new ExternalData { Assembly = assembly, Type = targetId, Member = member, Kind = method.MethodKind.ToString() };
                data.Operators.Add(key, external);
            }
            external.References++;
            if (external.Examples.Count < 3)
                external.Examples.Add(new Site(tree.FilePath, syntax.GetLocation().GetLineSpan().StartLinePosition.Line + 1));
        }).Visit(model.GetOperation(operationRoot));
    }
}

var roots = new SortedSet<string>(new[] { "SlimeController", "Updater", "DetPhysics", "Ability", "InstantAbility", "PlayerHandler", "GameSession" }, StringComparer.Ordinal);
var scriptTypes = new Dictionary<string, string[]>(StringComparer.Ordinal);
foreach (var entry in archive.Entries.Where(x => x.FullName.EndsWith(".cs.meta", StringComparison.Ordinal))) {
    using var reader = new StreamReader(entry.Open());
    var match = Regex.Match(reader.ReadToEnd(), @"(?m)^guid: ([0-9a-f]{32})\r?$");
    if (!match.Success) throw new Exception("Script GUID missing: " + entry.FullName);
    var sourcePath = entry.FullName[..^5];
    scriptTypes.Add(match.Groups[1].Value,
        types.Values.Where(x => x.Declarations.Any(d => d.File == sourcePath)).Select(x => x.Name).ToArray());
}
var serializedRoots = new SortedDictionary<string, SortedSet<string>>(StringComparer.Ordinal);
var serializedHashes = new SortedDictionary<string, string>(StringComparer.Ordinal);
foreach (var path in serializedPaths) {
    var entry = archive.GetEntry(path) ?? throw new Exception("Missing serialized source: " + path);
    using var stream = entry.Open();
    using var bytes = new MemoryStream();
    stream.CopyTo(bytes);
    var content = bytes.ToArray();
    var hash = Convert.ToHexString(SHA256.HashData(content)).ToLowerInvariant();
    if (sourceHashes.GetProperty(path).GetString() != hash) throw new Exception("Asset pin mismatch: " + path);
    serializedHashes.Add(path, hash);
    foreach (Match match in Regex.Matches(Encoding.UTF8.GetString(content), @"m_Script:\s*\{[^}\r\n]*guid:\s*([0-9a-f]{32})")) {
        if (!scriptTypes.TryGetValue(match.Groups[1].Value, out var names)) continue; // Package/engine scripts aren't source types.
        foreach (var name in names) {
            roots.Add(name);
            if (!serializedRoots.TryGetValue(name, out var paths)) {
                paths = new SortedSet<string>(StringComparer.Ordinal);
                serializedRoots.Add(name, paths);
            }
            paths.Add(path);
        }
    }
}
foreach (var (name, data) in types) {
    var ancestor = data.Base;
    var seen = new HashSet<string>();
    while (ancestor != null && seen.Add(ancestor)) {
        if (ancestor == "MonoUpdatable") { roots.Add(name); break; }
        ancestor = types.TryGetValue(ancestor, out var parent) ? parent.Base : null;
    }
}
foreach (var name in roots)
    if (!types.ContainsKey(name)) throw new Exception("Missing root: " + name);
var reachable = new SortedSet<string>(StringComparer.Ordinal);
var pending = new Stack<string>(roots);
while (pending.TryPop(out var name))
    if (reachable.Add(name)) foreach (var dependency in types[name].SourceTypes) pending.Push(dependency);
object Summary(IEnumerable<string> names) {
    var selected = names.Select(x => types[x]).ToArray();
    var external = selected.SelectMany(x => x.External.Values).ToArray();
    return new {
        types = selected.Length, declaration_lines = selected.Sum(x => x.Lines),
        external_assemblies = external.GroupBy(x => x.Assembly).OrderBy(x => x.Key, StringComparer.Ordinal)
            .Select(g => new { assembly = g.Key, types = g.Select(x => x.Type).Distinct().Count(),
                members = g.Select(x => x.Member).Distinct().Count(), references = g.Sum(x => x.References) }),
        unity_types = external.Where(x => x.Assembly.StartsWith("Unity", StringComparison.Ordinal))
            .GroupBy(x => x.Type).OrderByDescending(x => x.Sum(y => y.References)).ThenBy(x => x.Key, StringComparer.Ordinal)
            .Select(g => new { type = g.Key, members = g.Select(x => x.Member).Distinct().Count(), references = g.Sum(x => x.References) }),
        external_operators = selected.SelectMany(x => x.Operators.Values)
            .GroupBy(x => x.Assembly + ":" + x.Member).OrderBy(x => x.Key, StringComparer.Ordinal)
            .Select(g => new { assembly = g.First().Assembly, member = g.First().Member, references = g.Sum(x => x.References) }),
        features = selected.SelectMany(x => x.Features).GroupBy(x => x.Key).OrderBy(x => x.Key, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.Sum(x => x.Value))
    };
}
var output = new {
    schema = "bopl-source-audit-v1", source_commit = revision, files, serialized_files = serializedHashes,
    serialized_root_types = serializedRoots,
    reference_assemblies = assemblies.ToDictionary(x => Path.GetFileName(x)!,
        x => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(x))).ToLowerInvariant()),
    compiler_errors = diagnostics.Length,
    limitations = new[] {
        "Type-level static references overapproximate executed gameplay; presentation/menu/network dependencies are included.",
        "Asset script GUIDs contribute all types declared in their source file, conservatively including sibling/nested types.",
        "Persistent event callbacks, reflection, dynamically loaded assets and runtime dispatch can add dependencies absent from named source references.",
        "Roots include pinned serialized script references, all MonoUpdatable subclasses and seven central runtime classes; this is not proof of complete gameplay coverage.",
        "External member counts describe explicit name syntax. Bound operators/conversions in method/accessor bodies and field/property initializers are reported separately; neither is a complete shim contract.",
        "Declaration line counts include nested declarations in their parent and are not unique physical source lines."
    },
    all = Summary(types.Keys), root_types = roots, reachable_types = reachable,
    reachable = Summary(reachable), types
};
var destination = Path.GetFullPath(args[1]);
Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
await File.WriteAllTextAsync(destination, JsonSerializer.Serialize(output, new JsonSerializerOptions {
    WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
}) + "\n");
Console.WriteLine($"SOURCE_AUDIT_OK files={files.Count} types={types.Count} reachable={reachable.Count} compiler_errors=0 output={destination}");

record Site(string File, int Line);
sealed class TypeData {
    public required string Name { get; init; }
    public required string Kind { get; init; }
    public string? Base { get; init; }
    public required string[] Interfaces { get; init; }
    public int Lines { get; set; }
    public List<Site> Declarations { get; } = new();
    public SortedSet<string> SourceTypes { get; } = new(StringComparer.Ordinal);
    public SortedDictionary<string, ExternalData> External { get; } = new(StringComparer.Ordinal);
    public SortedDictionary<string, ExternalData> Operators { get; } = new(StringComparer.Ordinal);
    public SortedDictionary<string, int> Features { get; } = new(StringComparer.Ordinal);
}
sealed class OperatorVisitor(Action<IMethodSymbol, SyntaxNode> found) : OperationWalker {
    public override void Visit(IOperation? operation) {
        var method = operation switch {
            IBinaryOperation x => x.OperatorMethod,
            IUnaryOperation x => x.OperatorMethod,
            IConversionOperation x => x.OperatorMethod,
            IIncrementOrDecrementOperation x => x.OperatorMethod,
            ICompoundAssignmentOperation x => x.OperatorMethod,
            _ => null
        };
        if (method != null) found(method, operation!.Syntax);
        base.Visit(operation);
    }
}
sealed class ExternalData {
    public required string Assembly { get; init; }
    public required string Type { get; init; }
    public required string Member { get; init; }
    public required string Kind { get; init; }
    public int References { get; set; }
    public List<Site> Examples { get; } = new();
}
