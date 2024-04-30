// Listen for messages from content script
chrome.runtime.onMessage.addListener(async function (message, sender, sendResponse) {
    if (message.type === 'getSignLanguageVideo') {
        try {
            const videoUrl = await getStoredVideo(message.word);
            if (videoUrl) {
                sendResponse({ videoUrl });
            } else {
                const fetchedVideoUrl = await fetchSignLanguageVideo(message.word);
                sendResponse({ videoUrl: fetchedVideoUrl });
            }
        } catch (error) {
            console.error('Error fetching sign language video:', error);
            sendResponse({ error: 'Error fetching sign language video' });
        }
        return true;
    }
});

async function fetchSignLanguageVideo(word) {
    const url = `https://www.signasl.org/sign/${word}`;
    try {
        const response = await fetch(url);
        const html = await response.text();
        const soup = new DOMParser().parseFromString(html, "text/html");
        const videoTag = soup.querySelector('meta[itemprop="contentURL"]');
        if (!videoTag) {
            throw new Error("Video URL not found");
        }
        const videoUrl = videoTag.content;
        await storeVideo(word, videoUrl);

        return videoUrl;
    } catch (error) {
        throw new Error('Failed to fetch video: ' + error.message);
    }
}

async function storeVideo(word, videoUrl) {
    return new Promise((resolve, reject) => {
        chrome.storage.local.set({ [word]: videoUrl }, function () {
            if (chrome.runtime.lastError) {
                reject(chrome.runtime.lastError.message);
            } else {
                resolve();
            }
        });
    });
}

async function getStoredVideo(word) {
    return new Promise((resolve, reject) => {
        chrome.storage.local.get([word], function (result) {
            if (chrome.runtime.lastError) {
                reject(chrome.runtime.lastError.message);
            } else {
                resolve(result[word]);
            }
        });
    });
}
