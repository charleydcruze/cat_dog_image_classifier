console.log("Hi Charley :)")
//check whether browser support service workers
if ('serviceWorker' in navigator) {
  //wait until page loaded to avoid delaying rendering
  window.addEventListener('load', function () {
    //register service worker
    navigator.serviceWorker.register('serviceWorker.js').then(
      function (registration) {
        console.log('Service worker registration successful',
          registration);
      },
      function (err) {
        console.log('Service worker registration failed', err);
      });
  });
}

//set up cache name and files to add to it
const CACHE_NAME = 'Online Netcode';
const CACHE_URLS = ['index.html',];

  self.addEventListener("install", function (event) {
  console.log("Service worker installed");
  event.waitUntil(
    //create and open cache
    caches.open(CACHE_NAME)
      .then(function (cache) {
        console.log("Cache opened");
        //add all URLs to cache
        return cache.addAll(CACHE_URLS);
      })
  );
});

self.addEventListener('fetch', function (event) {
  event.respondWith(async function () {
    try {
      var res = await fetch(event.request);
      var cache = await caches.open('Online Netcode');
      cache.put(event.request.url, res.clone());
      return res;
    }
    catch (error) {
      return caches.match(event.request);
    }
  }());
});

self.addEventListener("install", (event) => {
    console.log("Service Worker : Installed!")

    event.waitUntil(
        
        (async() => {
            try {
                cache_obj = await caches.open(cache)
                cache_obj.addAll(caching_files)
            }
            catch{
                console.log("error occured while caching...")
            }
        })()
    )
} )

self.addEventListener('fetch', function (event) {
  event.respondWith(async function () {
    try {
      var res = await fetch(event.request);
      var cache = await caches.open('Online Netcode');
      cache.put(event.request.url, res.clone());
      return res;
    }
    catch (error) {
      return caches.match(event.request);
    }
  }());
});

/*
'script.js', 'GeoLocate.js','Battery.js','AddUser.js'   
                    
*/