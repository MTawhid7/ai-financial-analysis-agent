import { GoogleLogin, type CredentialResponse } from "@react-oauth/google";
import { useNavigate } from "@tanstack/react-router";
import { useAuth } from "../hooks/useAuth";

export function LoginPage() {
  const { signIn } = useAuth();
  const navigate = useNavigate();

  const handleSuccess = async (cred: CredentialResponse) => {
    if (!cred.credential) return;
    try {
      await signIn(cred.credential);
      navigate({ to: "/chat" });
    } catch {
      alert("Sign-in failed. Please try again.");
    }
  };

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-8 bg-zinc-950 px-4">
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-2 mb-6">
          <span className="text-3xl">📊</span>
        </div>
        <h1 className="text-3xl font-semibold text-zinc-50 tracking-tight">
          AI Financial Analyst
        </h1>
        <p className="text-zinc-400 text-sm max-w-xs">
          Conversational stock analysis powered by Gemini. Sign in to access
          your research history and personalised insights.
        </p>
      </div>

      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 flex flex-col items-center gap-4 w-full max-w-sm">
        <p className="text-zinc-300 text-sm font-medium">Continue with</p>
        {/* Explicit values for all optional props prevent the library from
            serialising `undefined` into the button iframe URL, which causes
            Google's server to return 403. */}
        <GoogleLogin
          onSuccess={handleSuccess}
          onError={() => alert("Google sign-in failed")}
          theme="filled_black"
          shape="pill"
          size="large"
          width="280"
          text="signin_with"
          logo_alignment="left"
        />
        <p className="text-zinc-500 text-xs text-center mt-2">
          Your data is stored locally. We don't share your information.
        </p>
      </div>
    </div>
  );
}
