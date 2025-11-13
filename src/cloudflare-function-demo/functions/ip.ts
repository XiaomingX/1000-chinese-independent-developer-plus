export const onRequest: PagesFunction = async (context) => {
  const ip = context.request.headers.get('CF-Connecting-IP');
  return new Response(ip);
};
