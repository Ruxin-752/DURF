interface Env {
  DB: D1Database;
  ADMIN_EXPORT_TOKEN?: string;
  SITE_ORIGIN?: string;
}

declare namespace Cloudflare {
  interface Env {
    DB: D1Database;
    ADMIN_EXPORT_TOKEN?: string;
    SITE_ORIGIN?: string;
  }
}
