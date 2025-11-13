type Env = unknown;
type Params = 'name';

/**
 * @see {@link https://developers.cloudflare.com/pages/platform/functions/routing/#dynamic-routes}
 * @see {@link https://developers.cloudflare.com/pages/platform/functions/api-reference/#eventcontext}
 */
export const onRequest: PagesFunction<Env, Params> = async (context) => {
  return new Response(`Hello, ${context.params.name}!`);
};
