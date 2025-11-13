/**
 * To bind Workers AI to your Pages Function, you can configure a Workers AI binding in wrangler.toml or the Cloudflare dashboard.
 *
 * @see {@link https://blog.cloudflare.com/workers-ai-streaming/}
 */

interface Env {
  AI: Ai;
}

export const onRequest: PagesFunction<Env> = async (context) => {
  const input = {
    prompt: "What is the origin of the phrase Hello, World",
    stream: true,
  };

  const stream = await context.env.AI.run(
    "@cf/meta/llama-3.1-8b-instruct",
    input,
  );

  return new Response(stream as BodyInit, {
    headers: { "content-type": "text/event-stream" },
  });
};
