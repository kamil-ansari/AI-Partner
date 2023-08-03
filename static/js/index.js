function getHardResponse(userText) {
    let url = 'http://localhost:5000/chat';
    let data = {
        userText: userText
    };

    fetch(url, {
        method: 'POST', 
        body: JSON.stringify(data), 
        headers:{
          'Content-Type': 'application/json'
        }
    }).then(res => res.json())
    .catch(error => console.error('Error:', error))
    .then(response => {
        let botHtml = '<p class="botText"><span>' + response.botResponse + '</span></p>';
        $("#chatbox").append(botHtml);
        document.getElementById("chat-bar-bottom").scrollIntoView(true);
    });
}
