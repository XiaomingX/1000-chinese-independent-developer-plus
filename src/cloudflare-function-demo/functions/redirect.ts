/**
 * @see {@link https://developers.cloudflare.com/workers/examples/redirect/}
 */
export const onRequest: PagesFunction = async (context) => {
  const url = new URL(context.request.url);
  const status = 301;
  return Response.redirect(
    `http://example.com${url.pathname}${url.search}`,
    status
  );
};
