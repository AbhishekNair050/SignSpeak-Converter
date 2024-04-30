// Listen for mouseover events on text elements
document.addEventListener('mouseover', function (event) {
    const target = event.target;
    if (target.nodeType === Node.TEXT_NODE && target.textContent.trim().length > 0) {
        const word = target.textContent.trim();
        chrome.runtime.sendMessage({ type: 'getSignLanguageVideo', word }, function (response) {
            if (response && response.videoUrl) {
                showSignLanguageVideo(response.videoUrl, event.clientX, event.clientY);
            } else {
                console.error('Failed to fetch sign language video:', response.error);
            }
        });
    }
});

function showSignLanguageVideo(videoUrl, x, y) {
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.top = `${y}px`;
    overlay.style.left = `${x}px`;
    overlay.style.padding = '10px';
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    overlay.style.color = '#fff';
    overlay.style.borderRadius = '5px';
    overlay.style.zIndex = '9999';
    overlay.style.pointerEvents = 'none';

    const video = document.createElement('video');
    video.src = videoUrl;
    video.controls = true;
    video.style.maxWidth = '100%';
    video.style.height = 'auto';
    overlay.appendChild(video);
    document.body.appendChild(overlay);
    setTimeout(() => {
        document.body.removeChild(overlay);
    }, 5000);
}
