using System;
using System.Text;
using UnityEngine;
using UnityEngine.InputSystem;

// Dev instrumentation: auto-spawned perf HUD + periodic log lines.
// Toggle overlay with F8. Logs a [PERF] line every 5s regardless.
// Sim tick timing is fed by Updater.TickSimulation via ReportSimTick.
public class PerfHud : MonoBehaviour
{
	private static PerfHud instance;

	private const int WindowSize = 240;

	private float[] frameTimes = new float[WindowSize];

	private int frameIdx;

	private int frameCount;

	private float logTimer;

	private float simMsAccum;

	private int simTickCount;

	private float simMsWindow;

	private int simTicksWindow;

	private long allocPrev;

	private float fpsSmoothed;

	private bool visible = true;

	private readonly StringBuilder sb = new StringBuilder(256);

	private GUIStyle style;

	[RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
	private static void Spawn()
	{
		if (instance == null)
		{
			GameObject gameObject = new GameObject("PerfHud");
			instance = gameObject.AddComponent<PerfHud>();
			UnityEngine.Object.DontDestroyOnLoad(gameObject);
		}
	}

	public static void ReportSimTick(float ms)
	{
		if (instance != null)
		{
			instance.simMsWindow += ms;
			instance.simTicksWindow++;
		}
	}

	private void Update()
	{
		float unscaledDeltaTime = Time.unscaledDeltaTime;
		frameTimes[frameIdx] = unscaledDeltaTime;
		frameIdx = (frameIdx + 1) % WindowSize;
		frameCount++;
		float num = 1f / Mathf.Max(unscaledDeltaTime, 1E-05f);
		fpsSmoothed = ((frameCount <= 1) ? num : Mathf.Lerp(fpsSmoothed, num, 0.03f));
		if (Keyboard.current != null && Keyboard.current.f8Key.wasPressedThisFrame)
		{
			visible = !visible;
		}
		logTimer += unscaledDeltaTime;
		if (logTimer >= 5f)
		{
			logTimer = 0f;
			float avg;
			float p95;
			GetStats(out avg, out p95);
			long num2 = GC.GetAllocatedBytesForCurrentThread();
			long num3 = ((allocPrev == 0L) ? 0 : (num2 - allocPrev));
			allocPrev = num2;
			Debug.Log($"[PERF] fps={1f / Mathf.Max(avg, 1E-06f):F1} frameAvg={avg * 1000f:F2}ms p95={p95 * 1000f:F2}ms simTickAvg={(simTicksWindow <= 0 ? 0f : simMsWindow / simTicksWindow):F3}ms simTicks={simTicksWindow} alloc5s={num3 / 1024}KB scene={UnityEngine.SceneManagement.SceneManager.GetActiveScene().name}");
			simMsWindow = 0f;
			simTicksWindow = 0;
		}
	}

	private void GetStats(out float avg, out float p95)
	{
		int num = Math.Min(frameCount, WindowSize);
		if (num == 0)
		{
			avg = 0f;
			p95 = 0f;
			return;
		}
		float[] array = new float[num];
		Array.Copy(frameTimes, array, num);
		Array.Sort(array);
		float num2 = 0f;
		for (int i = 0; i < num; i++)
		{
			num2 += array[i];
		}
		avg = num2 / num;
		p95 = array[Mathf.Min((int)(num * 0.95f), num - 1)];
	}

	private void OnGUI()
	{
		if (!visible)
		{
			return;
		}
		GetStats(out var avg, out var p95);
		if (style == null)
		{
			style = new GUIStyle(GUI.skin.label)
			{
				fontSize = 14
			};
			style.normal.textColor = Color.white;
		}
		GUI.Box(new Rect(5f, 5f, 330f, 78f), GUIContent.none);
		sb.Length = 0;
		sb.Append("FPS ").Append(fpsSmoothed.ToString("F1")).Append("  frame ").Append((avg * 1000f).ToString("F2")).Append("ms p95 ").Append((p95 * 1000f).ToString("F2")).Append("ms\n");
		sb.Append("sim tick avg ").Append((simTicksWindow <= 0 ? 0f : simMsWindow / simTicksWindow).ToString("F3")).Append("ms  ticks/s ~").Append((simTicksWindow / 5).ToString()).Append("\n");
		sb.Append("scene ").Append(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name).Append("  [F8] hide");
		GUI.Label(new Rect(10f, 8f, 320f, 72f), sb.ToString(), style);
	}
}
