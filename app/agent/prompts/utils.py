"""유틸리티 함수 - 날짜 관련"""

from datetime import datetime


def get_today_str() -> str:
    """오늘 날짜 문자열"""
    return datetime.now().strftime("%Y년 %m월 %d일")


def get_current_year() -> str:
    """현재 연도 문자열"""
    return datetime.now().strftime("%Y")


def get_current_month_year() -> str:
    """현재 월과 연도 문자열 (예: "December 2025")"""
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    now = datetime.now()
    return f"{month_names[now.month]} {now.year}"

