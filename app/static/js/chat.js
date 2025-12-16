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
        { icon: "🤖", text: "LLM", domain: "LLM", query: "LLM 서비스들을 비교해주세요" },
        { icon: "💻", text: "코딩 AI", domain: "코딩", query: "코딩 AI 도구들을 비교해주세요" },
        { icon: "🎨", text: "디자인 AI", domain: "디자인", query: "디자인 AI 서비스들을 비교해주세요" }
    ];
    
    options.forEach(option => {
        const btn = document.createElement("button");
        btn.className = "option-btn";
        btn.innerHTML = `<span class="icon">${option.icon}</span> ${option.text}`;
        btn.onclick = () => selectOption(option.domain, option.query);
        optionsContainer.appendChild(btn);
    });
    
    messages.appendChild(optionsContainer);
    messages.scrollTop = messages.scrollHeight;
}

// 현재 선택된 도메인 (전역 변수)
let currentDomain = "LLM";

// 선택지 클릭 시
async function selectOption(domain, query) {
    currentDomain = domain;
    
    // 선택 버튼들 숨기기
    const optionsContainer = document.getElementById("option-buttons-container");
    if (optionsContainer) {
        optionsContainer.style.display = 'none';
    }
    
    // 사용자가 선택한 것처럼 메시지 추가
    appendMessage("user", query);
    
    // 로딩 표시
    const loadingId = showLoading();
    
    try {
        // API 호출
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                message: query,
                domain: domain
            })
        });
        
        const data = await response.json();
        removeLoading(loadingId);
        appendMessage("assistant", data.reply);
    } catch (error) {
        console.error("Error:", error);
        removeLoading(loadingId);
        appendMessage("assistant", "⚠️ 오류가 발생했습니다. 다시 시도해주세요.");
    }
}

// 메시지 전송
async function sendMessage() {
    const input = document.getElementById("userInput");
    const message = input.value.trim();
    if (!message) return;
    
    // 사용자 메시지 추가
    appendMessage("user", message);
    input.value = "";
    
    // 로딩 표시
    const loadingId = showLoading();
    
    try {
        // API 호출 (현재 도메인 사용)
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                message: message,
                domain: currentDomain
            })
        });
        
        const data = await response.json();
        removeLoading(loadingId);
        appendMessage("assistant", data.reply);
    } catch (error) {
        console.error("Error:", error);
        removeLoading(loadingId);
        appendMessage("assistant", "⚠️ 오류가 발생했습니다. 다시 시도해주세요.");
    }
}

// 로딩 인디케이터 표시
function showLoading() {
    const messages = document.getElementById("messages");
    const loadingDiv = document.createElement("div");
    const loadingId = "loading-" + Date.now();
    loadingDiv.id = loadingId;
    loadingDiv.className = "message assistant loading";
    loadingDiv.innerHTML = `
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
        <p style="margin-top: 8px; color: #666; font-size: 13px;">
            🔍 AI 서비스 정보를 분석하고 있습니다...
        </p>
    `;
    messages.appendChild(loadingDiv);
    messages.scrollTop = messages.scrollHeight;
    return loadingId;
}

// 로딩 인디케이터 제거
function removeLoading(loadingId) {
    const loadingDiv = document.getElementById(loadingId);
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

// 메시지 추가
function appendMessage(role, text) {
    const messages = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = `message ${role}`;
    
    // HTML 및 마크다운 렌더링
    div.innerHTML = formatMarkdown(text);
    
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

// 간단한 마크다운 → HTML 변환
function formatMarkdown(text) {
    return text
        // 제목 변환
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
        
        // 굵은 글씨
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        
        // 링크
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
        
        // 리스트
        .replace(/^\- (.*$)/gim, '<li>$1</li>')
        
        // 구분선
        .replace(/^---$/gim, '<hr>')
        
        // 줄바꿈 처리 (연속된 줄바꿈은 무시, 단일 줄바꿈만 <br>)
        .replace(/\n{3,}/g, '<br>')  // 3개 이상 줄바꿈 → 1개 <br>
        .replace(/\n{2}/g, '<br>')   // 2개 줄바꿈 → 1개 <br>
        .replace(/\n/g, '');          // 단일 줄바꿈 → 삭제 (태그 간 자연스러운 간격)
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