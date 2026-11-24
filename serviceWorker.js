const CACHE_NAME = 'cat-dog-classifier-v1';

const ASSETS_TO_CACHE = [
    './',
    './index.html',
    './manifest.json',
    './serviceWorker.js',
    './CSS/style.css'
];

// Install Event - Cache all core static assets
self.addEventListener('install', (event) => {
    console.log('Service Worker: Installing...');

    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            console.log('Service Worker: Caching app assets');
            return cache.addAll(ASSETS_TO_CACHE);
        })
    );
});

// Activate Event - Clean up old caches
self.addEventListener('activate', (event) => {
    console.log('Service Worker: Activating...');

    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys
                    .filter((key) => key !== CACHE_NAME)
                    .map((key) => caches.delete(key))
            );
        })
    );
});

// Fetch Event - Serve from network, fall back to cache when offline
self.addEventListener('fetch', (event) => {
    event.respondWith(
        fetch(event.request)
            .then((response) => {
                // Clone and store fresh copies of fetched assets
                const resClone = response.clone();

                caches.open(CACHE_NAME).then((cache) => {
                    cache.put(event.request, resClone);
                });

                return response;
            })
            .catch(() => caches.match(event.request))
    );
});