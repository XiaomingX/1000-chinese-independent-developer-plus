export const onRequest: PagesFunction = async (context) => {
  const url = new URL(context.request.url);

  const params = {};

  url.searchParams.forEach((value, key) => {
    if (params[key] === undefined) {
      params[key] = value;
      return;
    }

    if (Array.isArray(params[key])) {
      params[key].push(value);
      return;
    }

    params[key] = [params[key], value];
  });

  return new Response(JSON.stringify(params), {
    headers: { 'content-type': 'application/json' },
  });
};
