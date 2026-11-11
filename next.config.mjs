/** @type {import('next').NextConfig} */
const docsBackendOrigin = (
  process.env.CPPT_DOCS_BACKEND_ORIGIN ?? "http://127.0.0.1:8010"
).replace(/\/+$/, "");

const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${docsBackendOrigin}/api/:path*` },
      { source: "/docs", destination: `${docsBackendOrigin}/docs` },
      { source: "/docs/:path*", destination: `${docsBackendOrigin}/docs/:path*` },
      { source: "/openapi.json", destination: `${docsBackendOrigin}/openapi.json` },
      { source: "/redoc", destination: `${docsBackendOrigin}/redoc` },
      { source: "/redoc/:path*", destination: `${docsBackendOrigin}/redoc/:path*` },
    ];
  },
};

export default nextConfig;
