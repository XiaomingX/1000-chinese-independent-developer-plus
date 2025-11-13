/**
 * @see {@link https://developers.cloudflare.com/pages/platform/functions/bindings/#kv-namespaces}
 */
interface Env {
  KV_BINDING: KVNamespace;
}

const html = (key, value) => `
<form method="post">
  <label>${key}:
    <input name="${key}" value="${value || ''}">
  </label>
  <input type="submit" value="Set">
</form>
`;

const KEY = 'KEY';

export const onRequestGet: PagesFunction<Env> = async (context) => {
  const value = await context.env.KV_BINDING.get(KEY);
  return new Response(html(KEY, value), {
    headers: { 'content-type': 'text/html' },
  });
};

export const onRequestPost: PagesFunction<Env> = async (context) => {
  const formData = await context.request.formData();
  const value = formData.get(KEY);
  await context.env.KV_BINDING.put(KEY, value);
  return new Response(html(KEY, value), {
    headers: { 'content-type': 'text/html' },
  });
};
