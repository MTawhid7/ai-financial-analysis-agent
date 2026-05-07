import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getMe, logout, signInWithGoogle, type UserProfile } from "../lib/api";

export function useAuth() {
  const queryClient = useQueryClient();

  const { data: user, isLoading, isError } = useQuery<UserProfile | null>({
    queryKey: ["me"],
    queryFn: async () => {
      try {
        return await getMe();
      } catch {
        return null;
      }
    },
    staleTime: 5 * 60 * 1000, // Re-validate every 5 minutes
    retry: false,
  });

  const signIn = async (idToken: string) => {
    const profile = await signInWithGoogle(idToken);
    queryClient.setQueryData(["me"], profile);
    return profile;
  };

  const signOut = async () => {
    await logout();
    queryClient.clear();
  };

  return {
    user: user ?? null,
    isLoading,
    isAuthenticated: !!user && !isError,
    signIn,
    signOut,
  };
}
