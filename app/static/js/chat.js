// 도메인은 항상 코딩으로 고정
let currentDomain = "코딩";

// 대화 이력 저장
let conversationHistory = [];

// localStorage 키
const STORAGE_KEY = 'ai_agent_conversation';

// 페이지 로드 시 대화 복원 또는 초기 환영 메시지
document.addEventListener('DOMContentLoaded', function() {
    // 저장된 대화 복원
    const savedConversation = localStorage.getItem(STORAGE_KEY);
    
    if (savedConversation) {
        try {
            const parsed = JSON.parse(savedConversation);
            conversationHistory = parsed.history || [];
            const savedMessages = parsed.messages || [];
            
            // 저장된 메시지 복원 (중복 제거)
            const messages = document.getElementById("messages");
            messages.innerHTML = '';
            
            const seenContents = new Set(); // 중복 체크용
            
            savedMessages.forEach(msg => {
                // 텍스트만 추출하여 중복 체크 (HTML 태그 제거)
                const textContent = msg.content.replace(/<[^>]+>/g, '').trim();
                const contentKey = `${msg.role}:${textContent}`;
                
                if (msg.content && textContent && !seenContents.has(contentKey)) {
                    const div = document.createElement("div");
                    div.className = `message ${msg.role}`;
                    
                    // HTML 플래그가 있으면 innerHTML로, 없으면 formatMarkdown 사용
                    if (msg.isHtml) {
                        div.innerHTML = msg.content;
                    } else {
                        div.innerHTML = formatMarkdown(msg.content);
                    }
                    
                    messages.appendChild(div);
                    seenContents.add(contentKey);
                }
            });
            
            messages.scrollTop = messages.scrollHeight;
            console.log('✅ 대화 복원 완료:', savedMessages.length, '개 메시지');
        } catch (e) {
            console.error('대화 복원 실패:', e);
            // 복원 실패 시 초기화
            localStorage.removeItem(STORAGE_KEY);
            setTimeout(() => {
                appendMessageWithOptions();
            }, 300);
        }
    } else {
        // 저장된 대화가 없으면 초기 환영 메시지 표시
        setTimeout(() => {
            appendMessageWithOptions();
        }, 300);
    }
});

// 초기 환영 메시지 추가
function appendMessageWithOptions() {
    const messages = document.getElementById("messages");
    
    // 이미 메시지가 있으면 환영 메시지 추가하지 않음
    if (messages.children.length > 0) {
        return;
    }
    
    // 메시지 말풍선
    const messageDiv = document.createElement("div");
    messageDiv.className = "message assistant";
    messageDiv.innerHTML = "안녕하세요! 👋<br><br><strong>코딩 AI 도입을 도와드리는 의사결정 어시스턴트</strong>입니다.<br><br>💬 <strong>\"우리 팀에 어떤 코딩 AI가 맞을까?\"</strong>처럼<br>편하게 질문해 주세요!<br><br>✨ 필요하면 제가 <strong>예산, 팀 규모, 보안 요구사항</strong> 등을<br>추가로 질문해서 <strong>맞춤 추천</strong>을 만들어 드립니다.";
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
    
    // localStorage에 저장 (환영 메시지 포함)
    saveConversation();
}

// 대화를 localStorage에 저장
function saveConversation() {
    try {
        const messages = document.getElementById("messages");
        const messageElements = Array.from(messages.children);
        
        // 화면의 메시지를 직접 저장 (중복 방지)
        const messagesToSave = [];
        const seenContents = new Set(); // 중복 체크용
        
        messageElements.forEach(el => {
            // 로딩 인디케이터는 제외
            if (el.classList.contains('loading')) {
                return;
            }
            
            const role = el.classList.contains('user') ? 'user' : 'assistant';
            
            // HTML 태그가 포함되어 있는지 확인
            const innerHTML = el.innerHTML || '';
            const textContent = el.textContent || el.innerText || '';
            const hasHtmlTags = /<[^>]+>/.test(innerHTML);
            
            // HTML 태그가 있으면 innerHTML 사용, 없으면 textContent 사용
            const content = hasHtmlTags ? innerHTML : textContent.trim();
            
            // 중복 체크: 동일한 내용이 이미 있으면 스킵
            const contentKey = `${role}:${textContent.trim()}`;
            if (content && !seenContents.has(contentKey)) {
                messagesToSave.push({ 
                    role, 
                    content: content,
                    isHtml: hasHtmlTags  // HTML 여부 플래그
                });
                seenContents.add(contentKey);
            }
        });
        
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
            history: conversationHistory,
            messages: messagesToSave,
            timestamp: Date.now()
        }));
    } catch (e) {
        console.error('대화 저장 실패:', e);
        // localStorage 용량 초과 시 오래된 대화 삭제
        if (e.name === 'QuotaExceededError') {
            console.warn('localStorage 용량 초과, 대화 초기화');
            localStorage.removeItem(STORAGE_KEY);
        }
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
    
    // 사용자 메시지를 이력에 추가
    conversationHistory.push({
        role: "user",
        content: message
    });
    
    // localStorage에 저장
    saveConversation();
    
    // 로딩 표시
    const loadingId = showLoading();
    
    try {
        // API 호출 (타임아웃: 180초)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 180000); // 180초 (3분)
        
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                message: message,
                domain: currentDomain,
                history: conversationHistory  // 대화 이력 전달
            }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        removeLoading(loadingId);
        
        // 배열이면 여러 메시지로, 문자열이면 하나의 메시지로
        if (Array.isArray(data.reply)) {
            // 여러 메시지를 순차적으로 추가
            data.reply.forEach((msg, index) => {
                setTimeout(() => {
                    appendMessage("assistant", msg);
                    // 마지막 메시지일 때만 저장
                    if (index === data.reply.length - 1) {
                        saveConversation();
                    }
                }, index * 500); // 0.5초 간격으로 추가
            });
            // AI 응답을 이력에 추가 (마지막 메시지만)
            conversationHistory.push({
                role: "assistant",
                content: data.reply[data.reply.length - 1]
            });
        } else {
            appendMessage("assistant", data.reply);
            // AI 응답을 이력에 추가
            conversationHistory.push({
                role: "assistant",
                content: data.reply
            });
            // localStorage에 저장
            saveConversation();
        }
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
            🔍 개인/팀 상황에 맞는 코딩 AI 도구를 분석하고 있습니다...
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
    
    // 대화 이력 초기화
    conversationHistory = [];
    
    // localStorage에서도 삭제
    localStorage.removeItem(STORAGE_KEY);
    
    // AI 환영 메시지 다시 표시
    setTimeout(() => {
        appendMessageWithOptions();
    }, 100);
}