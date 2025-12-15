// 페이지 로드 시 AI 환영 메시지
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        appendMessageWithOptions();
    }, 300);
});

// 선택지와 함께 메시지 추가
function appendMessageWithOptions() {
    const messages = document.getElementById("messages");
    
    // 메시지 말풍선 (줄바꿈 포함)
    const messageDiv = document.createElement("div");
    messageDiv.className = "message assistant";
    messageDiv.innerHTML = "안녕하세요! 👋 AI 서비스 분석 어시스턴트입니다.<br><br>분석하고 싶은 AI 서비스 분야를 선택해주세요.";
    messages.appendChild(messageDiv);
    
    // 선택 버튼들 (말풍선 밖에 별도로)
    const optionsContainer = document.createElement("div");
    optionsContainer.className = "option-buttons";
    optionsContainer.id = "option-buttons-container";
    
    const options = [
        { icon: "🤖", text: "LLM", query: "LLM 서비스들을 비교해주세요." },
        { icon: "💻", text: "코딩 AI", query: "코딩 AI 서비스들을을 비교해주세요." },
        { icon: "🎨", text: "디자인 AI", query: "디자인 AI 서비스들을 비교해주세요." }
    ];
    
    options.forEach(option => {
        const btn = document.createElement("button");
        btn.className = "option-btn";
        btn.innerHTML = `<span class="icon">${option.icon}</span> ${option.text}`;
        btn.onclick = () => selectOption(option.query);
        optionsContainer.appendChild(btn);
    });
    
    messages.appendChild(optionsContainer);
    messages.scrollTop = messages.scrollHeight;
}

// 선택지 클릭 시
function selectOption(query) {
    // 선택 버튼들 숨기기
    const optionsContainer = document.getElementById("option-buttons-container");
    if (optionsContainer) {
        optionsContainer.style.display = 'none';
    }
    
    // 사용자가 선택한 것처럼 메시지 추가
    appendMessage("user", query);
    
    // API 호출
    fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: query })
    })
    .then(response => response.json())
    .then(data => {
        appendMessage("assistant", data.reply);
    })
    .catch(error => {
        console.error("Error:", error);
        appendMessage("assistant", "⚠️ 오류가 발생했습니다. 다시 시도해주세요.");
    });
}

// 메시지 전송
async function sendMessage() {
    const input = document.getElementById("userInput");
    const message = input.value.trim();
    if (!message) return;
    
    // 사용자 메시지 추가
    appendMessage("user", message);
    input.value = "";
    
    try {
        // API 호출
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message })
        });
        
        const data = await response.json();
        appendMessage("assistant", data.reply);
    } catch (error) {
        console.error("Error:", error);
        appendMessage("assistant", "⚠️ 오류가 발생했습니다. 다시 시도해주세요.");
    }
}

// 메시지 추가
function appendMessage(role, text) {
    const messages = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = `message ${role}`;
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

// 대화 초기화
function clearChat() {
    const messages = document.getElementById("messages");
    messages.innerHTML = '';
    
    // AI 환영 메시지 다시 표시
    setTimeout(() => {
        appendMessageWithOptions();
    }, 100);
}