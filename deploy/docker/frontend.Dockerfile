ARG BASE_IMAGE=node:22-bookworm-slim
FROM ${BASE_IMAGE} AS dependencies
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM ${BASE_IMAGE} AS builder
WORKDIR /app
ARG NEXT_PUBLIC_API_BASE_URL=/api
ARG NEXT_PUBLIC_INFERENCE_BASE_URL=
ARG NEXT_PUBLIC_AUTH_CALLBACK_URL=/api/auth/callback
ARG CPPT_DOCS_BACKEND_ORIGIN=http://backend:8010
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL \
    NEXT_PUBLIC_INFERENCE_BASE_URL=$NEXT_PUBLIC_INFERENCE_BASE_URL \
    NEXT_PUBLIC_AUTH_CALLBACK_URL=$NEXT_PUBLIC_AUTH_CALLBACK_URL \
    CPPT_DOCS_BACKEND_ORIGIN=$CPPT_DOCS_BACKEND_ORIGIN \
    NEXT_TELEMETRY_DISABLED=1
COPY --from=dependencies /app/node_modules ./node_modules
COPY . .
# The repository does not require a public/ directory; create an empty one for the runtime copy.
RUN mkdir -p public
RUN npm run build

FROM ${BASE_IMAGE} AS runtime
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3100
WORKDIR /app
RUN groupadd --system --gid 1001 nodejs \
    && useradd --system --uid 1001 --gid nodejs nextjs
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
USER nextjs
EXPOSE 3100
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:3100/').then(r=>{if(!r.ok)process.exit(1)}).catch(()=>process.exit(1))"
CMD ["node", "server.js"]
