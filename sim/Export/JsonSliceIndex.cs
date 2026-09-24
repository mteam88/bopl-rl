using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace BoplSim.Export
{
	// A top-level JSON object ({"key": {...}, ...}) kept as its UTF-8 bytes plus a
	// key -> byte-range index, so one entry at a time is parsed into a Newtonsoft
	// tree on demand. assets.json is 37 MB on disk but ~300 MB as a JObject tree,
	// of which a few hundred entries are ever built; keeping the tree resident
	// cost every RL game process ~350 MB (a known gap).
	//
	// Parsing goes through the same JsonTextReader defaults JObject.Parse uses,
	// so the produced tokens are identical to a whole-file parse; only the line
	// info annotations are dropped (nothing reads them).
	internal sealed class JsonSliceIndex
	{
		private struct Slice
		{
			public int Start;
			public int Length;
		}

		private static readonly JsonLoadSettings LoadSettings = new JsonLoadSettings
		{
			LineInfoHandling = LineInfoHandling.Ignore,
		};

		private readonly byte[] bytes;
		private readonly Dictionary<string, Slice> slices = new Dictionary<string, Slice>();
		// Value of the string property named in Load(..., trackProperty) at the
		// top of each entry (assets.json: "path"), so callers can search by it
		// without parsing anything.
		internal readonly Dictionary<string, string> Tracked = new Dictionary<string, string>();

		private JsonSliceIndex(byte[] bytes)
		{
			this.bytes = bytes;
		}

		public int Count => slices.Count;
		public IEnumerable<string> Keys => slices.Keys;
		public bool ContainsKey(string key) => slices.ContainsKey(key);

		public static JsonSliceIndex Load(string file, string trackProperty = null)
		{
			var index = new JsonSliceIndex(File.ReadAllBytes(file));
			index.Scan(trackProperty);
			return index;
		}

		private void Scan(string trackProperty)
		{
			var reader = new Utf8JsonReader(bytes, isFinalBlock: true, state: default);
			if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
				throw new InvalidDataException("expected a top-level JSON object");
			while (reader.Read() && reader.TokenType == JsonTokenType.PropertyName)
			{
				string key = reader.GetString();
				if (!reader.Read()) throw new InvalidDataException("truncated JSON");
				int start = (int)reader.TokenStartIndex;
				string tracked = null;
				if (reader.TokenType == JsonTokenType.StartObject || reader.TokenType == JsonTokenType.StartArray)
				{
					int depth = reader.CurrentDepth;
					while (reader.Read())
					{
						if (reader.CurrentDepth == depth
							&& (reader.TokenType == JsonTokenType.EndObject || reader.TokenType == JsonTokenType.EndArray))
							break;
						if (trackProperty != null && tracked == null && reader.CurrentDepth == depth + 1
							&& reader.TokenType == JsonTokenType.PropertyName && reader.ValueTextEquals(trackProperty))
						{
							reader.Read();
							if (reader.TokenType == JsonTokenType.String) tracked = reader.GetString();
						}
					}
				}
				int end = (int)reader.BytesConsumed;
				slices[key] = new Slice { Start = start, Length = end - start };
				if (tracked != null) Tracked[key] = tracked;
			}
		}

		// Parse one entry; null when the key is absent. Each call builds a fresh
		// tree: callers keep what they build from it, not the tree.
		public JObject Parse(string key)
		{
			if (!slices.TryGetValue(key, out var slice)) return null;
			using (var stream = new MemoryStream(bytes, slice.Start, slice.Length, writable: false))
			using (var text = new StreamReader(stream, Encoding.UTF8))
			using (var reader = new JsonTextReader(text))
			{
				var result = JObject.Load(reader, LoadSettings);
				while (reader.Read()) { } // as JObject.Parse: trailing content throws
				return result;
			}
		}
	}
}
