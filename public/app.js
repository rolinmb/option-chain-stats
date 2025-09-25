async function fetchFileList(path) {
    try {
        const res = await fetch(path);
        if (!res.ok) {
            throw new Error(`Failed to fetch ${path}: ${res.status} ${res.statusText}`);
        }

        const text = await res.text();
        // Parse file links from directory listing
        const matches = [...text.matchAll(/href="([^"]+)"/g)];
        const files = matches.map(m => m[1]).filter(f => !f.startsWith("?") && !f.startsWith("/"));

        if (files.length === 0) {
            console.warn(`No files found at ${path}`);
        }
        return files;
    } catch (err) {
        console.error(err);
        return [];
    }
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

async function loadCSVs() {
    let select = document.getElementById("csv-select");
    select.innerHTML = "";
    const files = await fetchFileList("/data/");

    if (files.length === 0) {
        select.innerHTML = "<option>No CSV files found</option>";
        document.getElementById("csv-table").innerHTML = "<p>No CSV files available to display.</p>";
        return;
    }

    files.filter(f => f.endsWith(".csv")).forEach(f => {
        let opt = document.createElement("option");
        opt.value = f;
        opt.textContent = f;
        select.appendChild(opt);
    });

    select.addEventListener("change", () => showCSV(select.value));
    if (select.value) showCSV(select.value);
}

async function loadImages() {
    let select = document.getElementById("img-select");
    select.innerHTML = "";
    const files = await fetchFileList("/img/");

    if (files.length === 0) {
        select.innerHTML = "<option>No images found</option>";
        document.getElementById("preview").alt = "No images available";
        document.getElementById("preview").src = "";
        return;
    }

    files.filter(f => f.match(/\.(png|jpg|jpeg)$/i)).forEach(f => {
        let opt = document.createElement("option");
        opt.value = f;
        opt.textContent = f;
        select.appendChild(opt);
    });

    select.addEventListener("change", () => {
        document.getElementById("preview").src = "/img/" + select.value;
        document.getElementById("preview").alt = select.value;
    });

    if (select.value) {
        document.getElementById("preview").src = "/img/" + select.value;
        document.getElementById("preview").alt = select.value;
    }
}

loadCSVs();
loadImages();