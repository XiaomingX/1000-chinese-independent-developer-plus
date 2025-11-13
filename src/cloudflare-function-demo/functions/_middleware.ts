/**
 * @see {@link https://developers.cloudflare.com/pages/platform/functions/middleware/#add-middleware}
 */
async function errorHandling(context) {
  try {
    return await context.next();
  } catch (err) {
    return new Response(`${err.message}\n${err.stack}`, { status: 500 });
  }
}

export const onRequest = [errorHandling];
