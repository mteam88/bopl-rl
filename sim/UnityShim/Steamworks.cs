using System;
using System.Collections.Generic;
using System.Threading.Tasks;

// Facepunch.Steamworks stubs. Steam is never initialised in the sim; all
// online paths are dead (GameLobby.isOnlineGame is false under RL/replay).
namespace Steamworks
{
	using Steamworks.Data;

	public struct SteamId
	{
		public ulong Value;
		public static implicit operator ulong(SteamId id) => id.Value;
		public static implicit operator SteamId(ulong value) => new SteamId { Value = value };
		public bool IsValid => Value != 0;
		public uint AccountId => (uint)(Value & 0xFFFFFFFFul);
		public bool IsMe => false;
		public ulong Id => Value;
		public override string ToString() => Value.ToString();
	}

	public struct AppId
	{
		public uint Value;
		public static implicit operator uint(AppId id) => id.Value;
		public static implicit operator AppId(uint value) => new AppId { Value = value };
	}

	public static class SteamClient
	{
		public static bool IsValid => false;
		public static bool IsLoggedOn => false;
		public static SteamId SteamId => default;
		public static string Name => "SimPlayer";
		public static AppId AppId => default;
		public static void Init(uint appid, bool asyncCallbacks = true) => throw new InvalidOperationException("Steam unavailable in sim");
		public static void Shutdown() { }
		public static void RunCallbacks() { }
	}

	public struct Friend
	{
		public SteamId Id;
		public Friend(SteamId id) { Id = id; }
		public string Name => "SimFriend";
		public bool IsMe => false;
		public bool IsFriend => false;
		public bool IsOnline => false;
		public bool IsPlayingThisGame => false;
		public bool IsIn(SteamId groupOrRoom) => false;
		public Task<Steamworks.Data.Image?> GetMediumAvatarAsync() => Task.FromResult<Steamworks.Data.Image?>(null);
		public Task<Steamworks.Data.Image?> GetSmallAvatarAsync() => Task.FromResult<Steamworks.Data.Image?>(null);
	}

	public static class SteamFriends
	{
		public static event Action<Data.Lobby, SteamId> OnGameLobbyJoinRequested { add { } remove { } }
		public static IEnumerable<Friend> GetFriends() => Array.Empty<Friend>();
		public static void OpenOverlay(string type) { }
		public static void OpenUserOverlay(SteamId id, string type) { }
		public static void OpenGameInviteOverlay(SteamId lobby) { }
		public static Task<Steamworks.Data.Image?> GetMediumAvatarAsync(SteamId id) => Task.FromResult<Steamworks.Data.Image?>(null);
	}

	public static class SteamMatchmaking
	{
		public static event Action<Data.Result, Data.Lobby> OnLobbyCreated { add { } remove { } }
		public static event Action<Data.Lobby, Friend, string> OnChatMessage { add { } remove { } }
		public static event Action<Data.Lobby> OnLobbyEntered { add { } remove { } }
		public static event Action<Data.Lobby, Friend> OnLobbyMemberJoined { add { } remove { } }
		public static event Action<Data.Lobby, Friend> OnLobbyMemberLeave { add { } remove { } }
		public static event Action<Data.Lobby, Friend> OnLobbyMemberDisconnected { add { } remove { } }
		public static event Action<Data.Lobby> OnLobbyDataChanged { add { } remove { } }
		public static event Action<Data.Lobby, Friend> OnLobbyMemberDataChanged { add { } remove { } }
		public static event Action<Data.Lobby, uint, ushort, SteamId> OnLobbyGameCreated { add { } remove { } }
		public static event Action<Data.Lobby, SteamId> OnLobbyInvite { add { } remove { } }
		public static Task<Data.Lobby?> CreateLobbyAsync(int maxMembers = 100) => Task.FromResult<Data.Lobby?>(null);
		public static Task<Data.Lobby?> JoinLobbyAsync(SteamId lobbyId) => Task.FromResult<Data.Lobby?>(null);
		public static Data.LobbyQuery LobbyList => default;
	}

	public static class SteamNetworkingSockets
	{
		public static T CreateRelaySocket<T>(int virtualport = 0) where T : SocketManager, new() => new T();
		public static T ConnectRelay<T>(SteamId serverId, int virtualport = 0) where T : ConnectionManager, new() => new T();
		public static event Action<Data.Connection, Data.ConnectionInfo> OnConnectionStatusChanged { add { } remove { } }
	}

	public static class SteamNetworkingUtils
	{
		public static void InitRelayNetworkAccess() { }
		public static Data.NetPingLocation? LocalPingLocation => null;
	}

	public static class SteamUserStats
	{
		public static IEnumerable<Data.Achievement> Achievements => Array.Empty<Data.Achievement>();
		public static void StoreStats() { }
	}

	public static class SteamUser
	{
		public static SteamId SteamId => default;
	}

	public static class SteamApps
	{
		public static string AppLanguage => "english";
		public static string GameLanguage => "english";
		public static bool IsSubscribed => false;
	}

	public class SocketManager
	{
		public ISocketManager Interface { get; set; }
		public List<Data.Connection> Connected = new List<Data.Connection>();
		public Data.Socket Socket { get; set; }
		public virtual void OnConnecting(Data.Connection connection, Data.ConnectionInfo info) { }
		public virtual void OnConnected(Data.Connection connection, Data.ConnectionInfo info) { }
		public virtual void OnDisconnected(Data.Connection connection, Data.ConnectionInfo info) { }
		public virtual void OnMessage(Data.Connection connection, Data.NetIdentity identity, IntPtr data, int size, long messageNum, long recvTime, int channel) { }
		public void Receive(int bufferSize = 32) { }
		public bool Close() => true;
	}

	public interface ISocketManager
	{
		void OnConnecting(Data.Connection connection, Data.ConnectionInfo info);
		void OnConnected(Data.Connection connection, Data.ConnectionInfo info);
		void OnDisconnected(Data.Connection connection, Data.ConnectionInfo info);
		void OnMessage(Data.Connection connection, Data.NetIdentity identity, IntPtr data, int size, long messageNum, long recvTime, int channel);
	}

	public class ConnectionManager
	{
		public IConnectionManager Interface { get; set; }
		public Data.Connection Connection;
		public bool Connected { get; set; }
		public bool Connecting { get; set; }
		public virtual void OnConnecting(Data.ConnectionInfo info) { }
		public virtual void OnConnected(Data.ConnectionInfo info) { }
		public virtual void OnDisconnected(Data.ConnectionInfo info) { }
		public virtual void OnMessage(IntPtr data, int size, long messageNum, long recvTime, int channel) { }
		public void Receive(int bufferSize = 32) { }
		public void Close() { }
	}

	public interface IConnectionManager
	{
		void OnConnecting(Data.ConnectionInfo info);
		void OnConnected(Data.ConnectionInfo info);
		void OnDisconnected(Data.ConnectionInfo info);
		void OnMessage(IntPtr data, int size, long messageNum, long recvTime, int channel);
	}

	public enum NotificationPosition { TopLeft, TopRight, BottomLeft, BottomRight }

	public static class SteamUtils
	{
		public static NotificationPosition OverlayNotificationPosition { get; set; }
		public static bool IsOverlayEnabled => false;
	}
}

namespace Steamworks.Data
{
	public struct Lobby
	{
		public SteamId Id;
		public Lobby(SteamId id) { Id = id; }
		public int MemberCount => 0;
		public int MaxMembers { get; set; }
		public SteamId Owner
		{
			get => default;
			set { }
		}
		public IEnumerable<Friend> Members => Array.Empty<Friend>();
		public string GetData(string key) => "";
		public bool SetData(string key, string value) => true;
		public bool SetPublic() => true;
		public bool SetPrivate() => true;
		public bool SetFriendsOnly() => true;
		public bool SetJoinable(bool b) => true;
		public void Leave() { }
		public Task<RoomEnter> Join() => Task.FromResult(RoomEnter.Error);
		public bool InviteFriend(SteamId steamid) => false;
		public bool IsOwnedBy(SteamId steamid) => false;
		public bool SendChatString(string message) => false;
		public void SetGameServer(SteamId steamServer) { }
	}

	public enum RoomEnter { Success = 1, DoesntExist, NotAllowed, Full, Error, Banned, Limited, ClanDisabled, CommunityBan, MemberBlockedYou, YouBlockedMember, RatelimitExceeded }

	public struct LobbyQuery
	{
		public LobbyQuery WithMaxResults(int max) => this;
		public LobbyQuery WithKeyValue(string key, string value) => this;
		public LobbyQuery WithNotEqual(string key, int value) => this;
		public LobbyQuery WithEqual(string key, int value) => this;
		public LobbyQuery WithHigher(string key, int value) => this;
		public LobbyQuery WithLower(string key, int value) => this;
		public LobbyQuery WithSlotsAvailable(int minSlots) => this;
		public LobbyQuery FilterDistanceWorldwide() => this;
		public LobbyQuery FilterDistanceClose() => this;
		public LobbyQuery FilterDistanceFar() => this;
		public Task<Lobby[]> RequestAsync() => Task.FromResult<Lobby[]>(null);
	}

	public struct Connection
	{
		public uint Id;
		public bool Equals(Connection other) => Id == other.Id;
		public Result Close(bool linger = false, int reasonCode = 0, string debugString = "") => Result.OK;
		public Result SendMessage(byte[] data, SendType sendType = SendType.Reliable) => Result.OK;
		public Result SendMessage(byte[] data, int offset, int length, SendType sendType = SendType.Reliable) => Result.OK;
		public Result SendMessage(IntPtr ptr, int size, SendType sendType = SendType.Reliable) => Result.OK;
		public string ConnectionName { get; set; }
		public long UserData { get; set; }
		public override string ToString() => Id.ToString();
	}

	public struct ConnectionInfo
	{
		public SteamId Identity => default;
		public ConnectionState State => ConnectionState.None;
		public string EndReason => "";
		public string Address => "";
	}

	public enum ConnectionState { None, Connecting, FindingRoute, Connected, ClosedByPeer, ProblemDetectedLocally }

	[Flags]
	public enum SendType { Unreliable = 0, NoNagle = 1, NoDelay = 4, Reliable = 8 }

	public enum Result { None = 0, OK = 1, Fail = 2, NoConnection = 3, InvalidState = 11 }

	public struct NetIdentity
	{
		public SteamId SteamId => default;
		public static implicit operator NetIdentity(SteamId id) => new NetIdentity();
		public static implicit operator SteamId(NetIdentity identity) => default;
	}

	public struct Socket
	{
		public bool Close() => true;
	}

	public struct NetPingLocation { }

	public struct Achievement
	{
		public string Identifier => "";
		public bool State => false;
		public bool Trigger() => false;
		public void Clear() { }
	}

	public struct Image
	{
		public uint Width, Height;
		public byte[] Data;
		public Color GetPixel(int x, int y) => default;
	}

	public struct Color
	{
		public byte r, g, b, a;
	}
}
