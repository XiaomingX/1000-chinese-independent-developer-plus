/**
 * @see {@link https://developers.cloudflare.com/workers/runtime-apis/web-crypto/#methods}
 */
export const onRequest: PagesFunction = async (context) => {
  return new Response(crypto.randomUUID());
};
