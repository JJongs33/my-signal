// Service worker for "내 투자 신호": offline shell + weekly push notifications.
const CACHE = "signal-v1";
const SHELL = ["./", "index.html", "config.js", "manifest.json", "icon-192.png", "icon-512.png", "icon-180.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});

// data.json: always try the network first so the latest signal shows; fall back to the cached copy offline
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.pathname.endsWith("data.json")) {
    e.respondWith(fetch(e.request).then((r) => {
      const copy = r.clone();
      caches.open(CACHE).then((c) => c.put("data.json", copy));
      return r;
    }).catch(() => caches.match("data.json")));
    return;
  }
  e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request)));
});

self.addEventListener("push", (e) => {
  let d = { title: "이번 주 신호", body: "앱을 열어 확인하세요.", url: "./" };
  try { d = Object.assign(d, e.data.json()); } catch (_) {}
  e.waitUntil(self.registration.showNotification(d.title, { body: d.body, icon: "icon-192.png", badge: "icon-192.png", data: { url: d.url } }));
});

self.addEventListener("notificationclick", (e) => {
  e.notification.close();
  e.waitUntil(self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((list) => {
    for (const c of list) { if ("focus" in c) return c.focus(); }
    return self.clients.openWindow(e.notification.data && e.notification.data.url ? e.notification.data.url : "./");
  }));
});
