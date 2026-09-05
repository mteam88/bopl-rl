using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class ReferenceExport
{
    public static void Export()
    {
        string directory = Path.GetFullPath("native/reference-export");
        Directory.CreateDirectory(directory);
        var assets = new JObject();
        foreach (string guid in AssetDatabase.FindAssets("t:Prefab").Concat(AssetDatabase.FindAssets("t:ScriptableObject")).Distinct().OrderBy(s => s, StringComparer.Ordinal))
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            if (!path.StartsWith("Assets/", StringComparison.Ordinal)) continue;
            var root = AssetDatabase.LoadMainAssetAtPath(path);
            JToken data = root is GameObject gameObject ? Hierarchy(gameObject) : Properties(root);
            assets[guid] = new JObject { ["path"] = path, ["data"] = data };
        }
        Write(directory, "assets.json", assets);
        var scenes = new JArray();
        string[] paths = EditorBuildSettingsScene.GetActiveSceneList(EditorBuildSettings.scenes);
        for (int index = 6; index < paths.Length; index++)
        {
            var scene = EditorSceneManager.OpenScene(paths[index], OpenSceneMode.Single);
            string filename = "map-" + (index - 6) + ".json";
            var roots = new JArray(scene.GetRootGameObjects().Select(Hierarchy));
            Write(directory, filename, roots);
            scenes.Add(new JObject { ["level"] = index - 6, ["scene"] = scene.name,
                ["path"] = paths[index], ["file"] = filename });
        }
        Write(directory, "maps.json", scenes);
        var manifest = new JObject { ["schema"] = "bopl-assets-v1", ["reference"] = "67484d0", ["unity"] = Application.unityVersion };
        var hashes = new JObject();
        foreach (string path in Directory.GetFiles(directory, "*.json").Where(p => Path.GetFileName(p) != "manifest.json").OrderBy(p => p, StringComparer.Ordinal))
        {
            using var sha = SHA256.Create();
            hashes[Path.GetFileName(path)] = BitConverter.ToString(sha.ComputeHash(File.ReadAllBytes(path))).Replace("-", "").ToLowerInvariant();
        }
        manifest["files"] = hashes;
        Write(directory, "manifest.json", manifest);
        Debug.Log("REFERENCE_EXPORT_COMPLETE maps=" + scenes.Count + " assets=" + assets.Count);
    }

    private static JObject Hierarchy(GameObject root)
    {
        var components = new JArray();
        foreach (Component component in root.GetComponents<Component>())
            if (component != null) components.Add(Properties(component));
        var children = new JArray();
        for (int i = 0; i < root.transform.childCount; i++) children.Add(Hierarchy(root.transform.GetChild(i).gameObject));
        return new JObject { ["name"] = root.name, ["id"] = Id(root), ["layer"] = root.layer,
            ["active"] = root.activeSelf, ["components"] = components, ["children"] = children };
    }

    private static JObject Properties(UnityEngine.Object value)
    {
        var result = new JObject { ["type"] = value.GetType().FullName, ["id"] = Id(value) };
        var properties = new JObject();
        using var serialized = new SerializedObject(value);
        var property = serialized.GetIterator();
        while (property.Next(true))
        {
            JToken item;
            switch (property.propertyType)
            {
                case SerializedPropertyType.Generic: continue;
                case SerializedPropertyType.Integer: item = property.longValue; break;
                case SerializedPropertyType.Boolean: item = property.boolValue; break;
                case SerializedPropertyType.Float:
                    item = property.type == "double" ? new JObject { ["f64"] = BitConverter.DoubleToInt64Bits(property.doubleValue) }
                        : new JObject { ["f32"] = BitConverter.ToInt32(BitConverter.GetBytes(property.floatValue), 0) }; break;
                case SerializedPropertyType.String: item = property.stringValue; break;
                case SerializedPropertyType.ObjectReference: item = property.objectReferenceValue == null ? JValue.CreateNull() : new JValue(Id(property.objectReferenceValue)); break;
                case SerializedPropertyType.Enum: item = property.intValue; break;
                case SerializedPropertyType.LayerMask: item = property.intValue; break;
                case SerializedPropertyType.ArraySize: item = property.intValue; break;
                case SerializedPropertyType.AnimationCurve:
                    var curve = property.animationCurveValue;
                    item = new JObject { ["preWrap"] = (int)curve.preWrapMode, ["postWrap"] = (int)curve.postWrapMode,
                        ["keys"] = JToken.FromObject(curve.keys.Select(k => new {
                            time = ((BoplFixedMath.Fix)k.time).m_rawValue,
                            value = ((BoplFixedMath.Fix)k.value).m_rawValue,
                            inTangent = ((BoplFixedMath.Fix)k.inTangent).m_rawValue,
                            outTangent = ((BoplFixedMath.Fix)k.outTangent).m_rawValue })) }; break;
                default:
                    // Child scalar fields are enumerated by Next(true). Preserve a type tag
                    // for native presentation values that have no scalar children.
                    item = new JObject { ["serializedType"] = property.propertyType.ToString() }; break;
            }
            properties[property.propertyPath] = item;
        }
        result["properties"] = properties;
        return result;
    }

    private static string Id(UnityEngine.Object value) => GlobalObjectId.GetGlobalObjectIdSlow(value).ToString();
    private static void Write(string directory, string name, JToken value) => File.WriteAllText(Path.Combine(directory, name), value.ToString(Formatting.Indented) + "\n");
}
