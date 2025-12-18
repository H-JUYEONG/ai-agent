// 페이지 로드 시 AI 환영 메시지
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        appendMessageWithOptions();
    }, 300);
});

// 초기 환영 메시지 추가
function appendMessageWithOptions() {
    const messages = document.getElementById("messages");
    
    // 메시지 말풍선
    const messageDiv = document.createElement("div");
    messageDiv.className = "message assistant";
    messageDiv.innerHTML = "안녕하세요! 👋 코딩 AI 도입 의사결정 어시스턴트입니다.<br><br>팀 또는 회사의 상황을 알려주시면, 그에 맞는 코딩 AI 도구를 추천해드립니다.<br><br>다음 정보를 알려주세요:<br>• 💰 <strong>예산</strong> (예: 월 50만원 이하)<br>• 🔒 <strong>보안 요구사항</strong> (예: 코드가 외부로 유출되면 안 됨)<br>• 💻 <strong>사용하는 IDE</strong> (예: VS Code, IntelliJ, PyCharm)<br>• 📋 <strong>업무 특성</strong> (예: 웹 개발, 모바일 앱, 데이터 분석)";
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

// 도메인은 항상 코딩으로 고정
let currentDomain = "코딩";

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
        // API 호출 (타임아웃: 120초)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000); // 120초
        
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                message: message,
                domain: currentDomain
            }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        removeLoading(loadingId);
        appendMessage("assistant", data.reply);
    } catch (error) {
        console.error("Error:", error);
        removeLoading(loadingId);
        
        if (error.name === 'AbortError') {
            appendMessage("assistant", "⏱️ 요청 시간이 초과되었습니다. 질문을 단순화하여 다시 시도해주세요.");
        } else {
            appendMessage("assistant", "⚠️ 오류가 발생했습니다. 다시 시도해주세요.");
        }
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
            🔍 팀 상황에 맞는 코딩 AI 도구를 분석하고 있습니다...
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