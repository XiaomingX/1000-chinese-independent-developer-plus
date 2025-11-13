/**
 * @see {@link https://developers.cloudflare.com/workers/examples/return-json/}
 */
export const onRequest: PagesFunction = async (context) => {
  const data = {
    hello: 'world',
  };

  const json = JSON.stringify(data, null, 2);

  return new Response(json, {
    headers: {
      'content-type': 'application/json;charset=UTF-8',
    },
  });
};
