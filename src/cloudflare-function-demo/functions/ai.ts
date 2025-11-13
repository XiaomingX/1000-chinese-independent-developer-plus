/**
 * To bind Workers AI to your Pages Function, you can configure a Workers AI binding in wrangler.toml or the Cloudflare dashboard.
 *
 * @see {@link https://developers.cloudflare.com/pages/functions/bindings/#use-workers-ai-bindings}
 */

interface Env {
  AI: Ai;
}

export const onRequest: PagesFunction<Env> = async (context) => {
  const input = { prompt: "What is the origin of the phrase Hello, World" };

  const answer = await context.env.AI.run(
      "@cf/meta/llama-3.1-8b-instruct",
      input,
    );

  return Response.json(answer);
};
