/**
 * @see {@link https://developers.cloudflare.com/workers/runtime-apis/request/#incomingrequestcfproperties}
 * @see {@link https://developers.cloudflare.com/workers/examples/accessing-the-cloudflare-object/}
 */
export const onRequest: PagesFunction = async (context) => {
  return new Response(JSON.stringify(context.request.cf, null, 2), {
    headers: { 'content-type': 'application/json' },
  });
};
