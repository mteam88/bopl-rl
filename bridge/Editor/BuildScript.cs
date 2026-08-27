using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering;

public static class BuildScript
{
	public static void BuildLinux()
	{
		// Default to OpenGLCore: the Vulkan present path renders corrupted output
		// (black background / garbage tiles) over NVIDIA + Xwayland + mutter.
		PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneLinux64, new[] { GraphicsDeviceType.OpenGLCore });
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

	public static void BuildMac()
	{
		PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneOSX, new[] { GraphicsDeviceType.Metal });
#if UNITY_EDITOR_OSX
		UnityEditor.OSXStandalone.UserBuildSettings.architecture = OSArchitecture.ARM64;
#endif
		BuildPlayerOptions options = new BuildPlayerOptions
		{
			scenes = EditorBuildSettingsScene.GetActiveSceneList(EditorBuildSettings.scenes),
			locationPathName = "Builds/macOS/BoplBattle.app",
			target = BuildTarget.StandaloneOSX,
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
