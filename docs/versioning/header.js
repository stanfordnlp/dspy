const desktop = window.matchMedia("(min-width: 60em)");

function placeVersionPicker() {
    const picker = document.querySelector(".md-version");
    const search = document.querySelector(".md-search");
    const title = document.querySelector(".md-header__topic");
    if (!picker || !search || !title) return false;

    if (desktop.matches) {
        search.after(picker);
    } else {
        title.append(picker);
    }
    return true;
}

if (!placeVersionPicker()) {
    const observer = new MutationObserver(() => {
        if (placeVersionPicker()) observer.disconnect();
    });
    observer.observe(document.body, { childList: true, subtree: true });
}

desktop.addEventListener("change", placeVersionPicker);
