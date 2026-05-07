import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getMe, logout, signInWithGoogle, type UserProfile } from "../lib/api";

export function useAuth() {
  const queryClient = useQueryClient();

  const { data: user, isLoading } = useQuery<UserProfile | null>({
    queryKey: ["me"],
    queryFn: async () => {
      try {
        return await getMe();
      } catch {
        // 401 is expected on initial load and immediately after logout.
        // The catch returns null so the app stays on the login page.
        return null;
      }
    },
    staleTime: 5 * 60 * 1000,
    retry: false,
  });

  const signIn = async (idToken: string) => {
    const profile = await signInWithGoogle(idToken);
    // Populate the cache directly — avoids a round-trip GET /auth/me.
    queryClient.setQueryData(["me"], profile);
    return profile;
  };

  const signOut = async () => {
    await logout();
    // Set to null immediately so the UI shows the login page without waiting
    // for a network round-trip.  Do NOT call queryClient.clear() — that would
    // trigger an immediate re-fetch of every query (including /auth/me),
    // causing a visible 401 in the console on every logout.
    queryClient.setQueryData(["me"], null);
    queryClient.removeQueries({ queryKey: ["conversations"] });
    queryClient.removeQueries({ queryKey: ["preferences"] });
    queryClient.removeQueries({ queryKey: ["summaries"] });
  };

  return {
    user: user ?? null,
    isLoading,
    isAuthenticated: !!user,
    signIn,
    signOut,
  };
}
