// Service worker for "쫑우 투자신호": offline fallback + push notifications.
// Strategy: always try the network first (so a new version of the app and fresh data show up right away),
// and use the saved copy only when the phone is offline.
const CACHE = "signal-v34";
const SHELL = ["./", "index.html", "config.js", "manifest.json", "icon-192.png", "icon-512.png", "icon-180.png", "data.json", "prices.json"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL.map((u) => new Request(u, { cache: "reload" })))).catch(() => {}).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  e.respondWith(
    fetch(req, { cache: "no-store" }).then((res) => {
      if (res && res.ok) {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(req, copy));
      }
      return res;
    }).catch(() => caches.match(req, { ignoreSearch: true }).then((hit) => hit || caches.match("index.html")))
  );
});

self.addEventListener("push", (e) => {
  let d = { title: "쫑우 투자신호", body: "앱을 열어 확인하세요.", url: "./" };
  try { d = Object.assign(d, e.data.json()); } catch (_) {}
  // a new tag for every message: replacing an old notification with the same tag can be silent on Android
  e.waitUntil(self.registration.showNotification(d.title, { body: d.body, icon: "icon-192.png", badge: "icon-192.png", tag: "signal-" + Date.now(), data: { url: d.url } }));
});

self.addEventListener("notificationclick", (e) => {
  e.notification.close();
  e.waitUntil(self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((list) => {
    for (const c of list) { if ("focus" in c) return c.focus(); }
    return self.clients.openWindow(e.notification.data && e.notification.data.url ? e.notification.data.url : "./");
  }));
});
