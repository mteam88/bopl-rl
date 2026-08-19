using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class BuildScript
{
	public static void BuildLinux()
	{
		BuildPlayerOptions options = new BuildPlayerOptions
		{
			scenes = EditorBuildSettingsScene.GetActiveSceneList(EditorBuildSettings.scenes),
			locationPathName = "Builds/Linux/BoplBattle.x86_64",
			target = BuildTarget.StandaloneLinux64,
			options = BuildOptions.None
		};
		BuildReport report = BuildPipeline.BuildPlayer(options);
		BuildSummary summary = report.summary;
		Debug.Log("BUILD RESULT: " + summary.result + " errors=" + summary.totalErrors + " size=" + summary.totalSize);
		if (summary.result != BuildResult.Succeeded)
		{
			EditorApplication.Exit(1);
		}
	}
}
