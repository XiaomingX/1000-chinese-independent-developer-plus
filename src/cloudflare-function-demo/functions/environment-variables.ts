/**
 * @see {@link https://developers.cloudflare.com/pages/platform/functions/bindings/#environment-variables}
 */

enum Environment {
  development = 'development',
  staging = 'staging',
  production = 'production',
}

interface Env {
  ENVIRONMENT: Environment;
}

export const onRequest: PagesFunction<Env> = async (context) => {
  const environment = context.env.ENVIRONMENT;

  switch (environment) {
    case Environment.production:
      return new Response(`This is a ${environment} environment`);

    case Environment.staging:
      return new Response(`This is a ${environment} environment`);

    case Environment.development:
    default:
      return new Response(`This is a ${environment} environment`);
  }
};
