/**
 * @see {@link https://developers.cloudflare.com/pages/platform/functions/typescript/}
 */
interface Env {}

export const onRequest: PagesFunction<Env> = async (context) => {
  return new Response('Hello, TypeScript!');
};
