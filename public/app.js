async function fetchFileList(path) {
      // Uses directory listing if Flask or http.server exposes it.
      let res = await fetch(path);
      let text = await res.text();

      // Parse file links out of the directory listing (simple regex)
      let matches = [...text.matchAll(/href="([^"]+)"/g)];
      return matches.map(m => m[1]).filter(f => !f.startsWith("?") && !f.startsWith("/"));
    }

    async function loadCSVs() {
      let select = document.getElementById("csv-select");
      let files = await fetchFileList("/data/");
      files.filter(f => f.endsWith(".csv")).forEach(f => {
        let opt = document.createElement("option");
        opt.value = f;
        opt.textContent = f;
        select.appendChild(opt);
      });

      select.addEventListener("change", () => showCSV(select.value));
      if (select.value) showCSV(select.value);
    }

    async function showCSV(filename) {
      let res = await fetch("/data/" + filename);
      let text = await res.text();

      let parsed = Papa.parse(text.trim(), { header: false });

      let rows = parsed.data;
      let table = "<table><thead><tr>" +
      rows[0].map(h => `<th>${h}</th>`).join("") +
      "</tr></thead><tbody>" +
      rows.slice(1).map(r => "<tr>" + r.map(c => `<td>${c}</td>`).join("") + "</tr>").join("") +
      "</tbody></table>";

      document.getElementById("csv-table").innerHTML = table;
    }

    async function loadImages() {
      let select = document.getElementById("img-select");
      let files = await fetchFileList("/img/");
      files.filter(f => f.match(/\.(png|jpg|jpeg)$/)).forEach(f => {
        let opt = document.createElement("option");
        opt.value = f;
        opt.textContent = f;
        select.appendChild(opt);
      });

      select.addEventListener("change", () => {
        document.getElementById("preview").src = "/img/" + select.value;
      });

      if (select.value) document.getElementById("preview").src = "/img/" + select.value;
    }

    loadCSVs();
    loadImages();