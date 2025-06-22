let currentTabId = null;
let startTime = null;

function saveTimeSpent(tabId, endTime) {
  if (currentTabId === null || startTime === null) return;

  const timeSpent = Math.round((endTime - startTime) / 1000); // in seconds

  chrome.tabs.get(currentTabId, function(tab) {
    if (chrome.runtime.lastError || !tab || !tab.url.startsWith("http")) return;

    const record = {
      url: tab.url,
      title: tab.title || "",
      visitTime: new Date(startTime).toISOString(),
      timeSpent: timeSpent  // in seconds
    };

    chrome.storage.local.get({ history: [] }, function(data) {
      const historyList = data.history;
      historyList.push(record);
      chrome.storage.local.set({ history: historyList });
    });
  });
}

// When a tab becomes active
chrome.tabs.onActivated.addListener(activeInfo => {
  const endTime = Date.now();
  saveTimeSpent(currentTabId, endTime);  // Save previous tab's time
  currentTabId = activeInfo.tabId;
  startTime = Date.now();                // Start new timer
});

// When the URL of a tab changes (navigation)
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.active) {
    const endTime = Date.now();
    saveTimeSpent(currentTabId, endTime);
    currentTabId = tabId;
    startTime = Date.now();
  }
});

// Handle browser close or extension shutdown
chrome.runtime.onSuspend.addListener(() => {
  saveTimeSpent(currentTabId, Date.now());
});
