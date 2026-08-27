using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

public static class AssetAudit
{
	public static void Run()
	{
		StringBuilder sb = new StringBuilder();
		string[] all = AssetDatabase.GetAllAssetPaths();
		int checked_ = 0, bad = 0;
		foreach (string path in all)
		{
			if (!path.StartsWith("Assets/") || Directory.Exists(path))
			{
				continue;
			}
			if (path.EndsWith(".cs") || path.EndsWith(".cginc") || path.EndsWith(".txt") || path.EndsWith(".json") || path.EndsWith(".md") || path.EndsWith(".dll") || path.EndsWith(".pdf"))
			{
				continue;
			}
			checked_++;
			Object main = AssetDatabase.LoadMainAssetAtPath(path);
			if (main == null)
			{
				bad++;
				sb.AppendLine("[AUDIT-NULL] " + path);
				continue;
			}
			// Sprites whose backing texture failed
			if (main is Sprite s && s.texture == null)
			{
				bad++;
				sb.AppendLine("[AUDIT-NOTEX] " + path);
			}
			// Materials with missing shader
			if (main is Material m && (m.shader == null || m.shader.name == "Hidden/InternalErrorShader"))
			{
				bad++;
				sb.AppendLine("[AUDIT-BADSHADER] " + path + " shader=" + (m.shader == null ? "NULL" : m.shader.name));
			}
			if ((checked_ % 5000) == 0)
			{
				Debug.Log("[AUDIT] progress " + checked_);
			}
		}
		sb.AppendLine("[AUDIT] checked=" + checked_ + " bad=" + bad);
		File.WriteAllText("/tmp/asset-audit.txt", sb.ToString());
		Debug.Log("[AUDIT] done, checked=" + checked_ + " bad=" + bad);
		EditorApplication.Exit(0);
	}
}
