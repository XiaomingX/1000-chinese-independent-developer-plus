/**
 * @see {@link https://developers.cloudflare.com/pages/platform/functions/bindings/#secrets}
 */
interface Env {
  SECRET: string;
}

export const onRequest: PagesFunction<Env> = async (context) => {
  const secret = context.env.SECRET;
  return new Response(`SECRET: ${secret}`);
};
