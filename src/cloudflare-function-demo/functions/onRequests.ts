/**
 * @see {@link https://developers.cloudflare.com/pages/platform/functions/api-reference/#methods}
 */

/**
 * This function will be invoked on all requests no matter the request method.
 */
export const onRequest: PagesFunction = async (context) => {
  return new Response(`onRequest: ${context.request.method}`);
};

/**
 * This function will be invoked on all GET requests.
 */
export const onRequestGet: PagesFunction = async (context) => {
  return new Response(`onRequestGet: ${context.request.method}`);
};

/**
 * This function will be invoked on all POST requests.
 */
export const onRequestPost: PagesFunction = async (context) => {
  return new Response(`onRequestPost: ${context.request.method}`);
};

/**
 * This function will be invoked on all PATCH requests.
 */
export const onRequestPatch: PagesFunction = async (context) => {
  return new Response(`onRequestPatch: ${context.request.method}`);
};

/**
 * This function will be invoked on all PUT requests.
 */
export const onRequestPut: PagesFunction = async (context) => {
  return new Response(`onRequestPut: ${context.request.method}`);
};

/**
 * This function will be invoked on all DELETE requests.
 */
export const onRequestDelete: PagesFunction = async (context) => {
  return new Response(`onRequestDelete: ${context.request.method}`);
};

/**
 * This function will be invoked on all HEAD requests.
 */
export const onRequestHead: PagesFunction = async (context) => {
  return new Response(`onRequestHead: ${context.request.method}`);
};

/**
 * This function will be invoked on all OPTIONS requests.
 */
export const onRequestOptions: PagesFunction = async (context) => {
  return new Response(`onRequestOptions: ${context.request.method}`);
};
