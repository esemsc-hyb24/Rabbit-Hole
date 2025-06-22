document.getElementById("show").addEventListener("click", function () {
    chrome.storage.local.get("history", function (data) {
      const output = document.getElementById("output");
      output.innerHTML = "";
      data.history.slice().reverse().forEach((item) => {
        const li = document.createElement("li");
  
        li.innerHTML = `
          <div class="url">${item.title || item.url}</div>
          <div class="meta">
            ${item.url}<br>
            Visited: ${new Date(item.visitTime).toLocaleString()}<br>
            Time spent: ${item.timeSpent || 0} seconds
          </div>
        `;
  
        output.appendChild(li);
      });
    });
  });
  
  document.getElementById("clear").addEventListener("click", function () {
    chrome.storage.local.set({ history: [] }, function () {
      document.getElementById("output").innerHTML = "";
      alert("History cleared.");
    });
  });
  