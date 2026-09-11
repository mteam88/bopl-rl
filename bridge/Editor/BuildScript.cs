using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering;

public static class BuildScript
{
	public static void BuildLinux()
	{
		PlayerSettings.SetScriptingBackend(BuildTargetGroup.Standalone, ScriptingImplementation.Mono2x);
		BuildLinuxPlayer("Builds/Linux/BoplBattle.x86_64");
	}

	// IL2CPP variant of the same player (autoresearch: Matthew's 2026-09-07
	// priority). Same scenes and graphics API; the scripting backend is the only
	// difference. Requires the Linux IL2CPP editor module and the
	// com.unity.toolchain.linux-x86_64 package (Packages/manifest.json).
	// BOPL_IL2CPP_CONFIG selects Release (default) or Master.
	public static void BuildLinuxIL2CPP()
	{
		PlayerSettings.SetScriptingBackend(BuildTargetGroup.Standalone, ScriptingImplementation.IL2CPP);
		string config = System.Environment.GetEnvironmentVariable("BOPL_IL2CPP_CONFIG") ?? "Release";
		PlayerSettings.SetIl2CppCompilerConfiguration(BuildTargetGroup.Standalone,
			config == "Master" ? Il2CppCompilerConfiguration.Master : Il2CppCompilerConfiguration.Release);
		string output = System.Environment.GetEnvironmentVariable("BOPL_BUILD_OUTPUT") ?? "Builds/LinuxIL2CPP/BoplBattle.x86_64";
		BuildLinuxPlayer(output);
	}

	static void BuildLinuxPlayer(string location)
	{
		// Default to OpenGLCore: the Vulkan present path renders corrupted output
		// (black background / garbage tiles) over NVIDIA + Xwayland + mutter.
		PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneLinux64, new[] { GraphicsDeviceType.OpenGLCore });
		BuildPlayerOptions options = new BuildPlayerOptions
		{
			scenes = EditorBuildSettingsScene.GetActiveSceneList(EditorBuildSettings.scenes),
			locationPathName = location,
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
		// Mono like the Linux player (the in-game agent shim is a DllImport'ed dylib; the
		// synced ProjectSettings may carry IL2CPP from BuildLinuxIL2CPP, which the Mac editor lacks).
		PlayerSettings.SetScriptingBackend(BuildTargetGroup.Standalone, ScriptingImplementation.Mono2x);
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
