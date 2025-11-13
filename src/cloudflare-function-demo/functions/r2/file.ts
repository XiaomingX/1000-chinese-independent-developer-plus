import { Env, KEY } from './shared';

/**
 * @see {@link https://developers.cloudflare.com/workers/learning/using-streams/}
 */
export const onRequestGet: PagesFunction<Env> = async (context) => {
  const obj = await context.env.R2_BUCKET.get(KEY);
  const fileData = await obj.arrayBuffer();

  return new Response(fileData, {
    headers: {
      'Content-Disposition': 'attachment',
      'Access-Control-Expose-Headers': 'X-Custom-Metadata',
      'X-Custom-Metadata': JSON.stringify(obj.customMetadata),
    },
  });
};
