interface Env {
  DB: D1Database;
  ADMIN_EXPORT_TOKEN?: string;
}

declare namespace Cloudflare {
  interface Env {
    DB: D1Database;
    ADMIN_EXPORT_TOKEN?: string;
  }
}
