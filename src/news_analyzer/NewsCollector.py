"""
뉴스 수집 모듈
전세계 암호화폐 관련 뉴스를 수집하는 안정적인 시스템
"""

import feedparser
import requests
from bs4 import BeautifulSoup
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import yaml
import hashlib
from retrying import retry
import timeout_decorator

@dataclass
class NewsArticle:
    title: str
    content: str
    url: str
    published_time: datetime
    source: str
    language: str
    reliability_score: float
    article_hash: str

class NewsCollector:
    """
    뉴스 수집 및 전처리를 담당하는 클래스
    - RSS 피드 기반 뉴스 수집
    - 중복 제거 및 품질 검증
    - 안정적인 오류 처리
    """

    def __init__(self, config_path: str):
        # 설정 파일 로드
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)

        self.news_sources = self.config['news_sources']
        self.analysis_config = self.config['analysis_config']
        self.collection_config = self.config['collection_config']
        self.error_config = self.config['error_handling']

        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        self._SetupLogging()

        # 수집된 기사 저장 (중복 방지용)
        self.collected_articles: Dict[str, NewsArticle] = {}
        self.last_collection_time = {}

        self.logger.info("NewsCollector 초기화 완료")

    def _SetupLogging(self):
        """로깅 시스템 설정"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    @timeout_decorator.timeout(30)
    def _FetchRSSFeed(self, source_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """RSS 피드에서 뉴스 가져오기 (재시도 및 타임아웃 적용)"""
        try:
            rss_url = source_info['rss_feed']
            self.logger.info(f"{source_info['name']} RSS 피드 수집 시작: {rss_url}")

            feed = feedparser.parse(rss_url)

            if feed.bozo:
                self.logger.warning(f"RSS 피드 파싱 경고: {source_info['name']}")

            articles = []
            for entry in feed.entries[:self.collection_config['max_articles_per_source']]:
                try:
                    # 발행 시간 파싱
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_time = datetime(*entry.published_parsed[:6])
                    else:
                        published_time = datetime.now()

                    # 24시간 이내 기사만 수집
                    if datetime.now() - published_time > timedelta(
                        hours=self.collection_config['article_max_age_hours']
                    ):
                        continue

                    article_data = {
                        'title': entry.title,
                        'url': entry.link,
                        'published_time': published_time,
                        'source': source_info['name'],
                        'language': source_info['language'],
                        'reliability_score': source_info['reliability']
                    }

                    # 본문 내용 추출 시도
                    content = self._ExtractArticleContent(entry.link)
                    if content and len(content) >= self.collection_config['min_article_length']:
                        article_data['content'] = content
                        articles.append(article_data)

                except Exception as e:
                    self.logger.error(f"기사 처리 중 오류: {e}")
                    continue

            self.logger.info(f"{source_info['name']}: {len(articles)}개 기사 수집 완료")
            return articles

        except Exception as e:
            self.logger.error(f"RSS 피드 수집 실패 - {source_info['name']}: {e}")
            raise

    @timeout_decorator.timeout(15)
    def _ExtractArticleContent(self, url: str) -> Optional[str]:
        """기사 본문 내용 추출"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # 일반적인 기사 본문 태그들 시도
            content_selectors = [
                'article p',
                '.article-content p',
                '.post-content p',
                '.entry-content p',
                '.content p',
                'main p'
            ]

            content_text = ""
            for selector in content_selectors:
                paragraphs = soup.select(selector)
                if paragraphs:
                    content_text = ' '.join([p.get_text().strip() for p in paragraphs])
                    break

            # 최소 길이 검증
            if len(content_text) < self.collection_config['min_article_length']:
                return None

            return content_text

        except Exception as e:
            self.logger.debug(f"본문 추출 실패 - {url}: {e}")
            return None

    def _CreateArticleHash(self, title: str, content: str) -> str:
        """기사 중복 확인을 위한 해시 생성"""
        combined_text = f"{title}{content}"
        return hashlib.md5(combined_text.encode('utf-8')).hexdigest()

    def _IsDuplicate(self, article_data: Dict[str, Any]) -> bool:
        """중복 기사 확인"""
        title = article_data['title']
        content = article_data.get('content', '')
        article_hash = self._CreateArticleHash(title, content)

        # 기존에 수집된 기사와 비교
        if article_hash in self.collected_articles:
            return True

        # 유사도 기반 중복 검사 (간단한 제목 비교)
        for existing_article in self.collected_articles.values():
            if self._CalculateSimilarity(title, existing_article.title) > \
               self.collection_config['duplicate_check_similarity']:
                return True

        return False

    def _CalculateSimilarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간의 유사도 계산 (간단한 방식)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def CollectAllNews(self) -> List[NewsArticle]:
        """모든 뉴스 소스에서 뉴스 수집"""
        all_articles = []
        current_time = time.time()

        # 글로벌 소스 수집
        for source_name, source_info in self.news_sources['global'].items():
            try:
                # 업데이트 빈도 확인
                last_update = self.last_collection_time.get(source_name, 0)
                if current_time - last_update < source_info['update_frequency']:
                    continue

                articles = self._FetchRSSFeed(source_info)
                for article_data in articles:
                    if not self._IsDuplicate(article_data):
                        article = self._CreateNewsArticle(article_data)
                        all_articles.append(article)
                        self.collected_articles[article.article_hash] = article

                self.last_collection_time[source_name] = current_time

            except Exception as e:
                self.logger.error(f"소스 {source_name} 수집 실패: {e}")
                continue

        # 한국 소스 수집
        for source_name, source_info in self.news_sources['korean'].items():
            try:
                last_update = self.last_collection_time.get(source_name, 0)
                if current_time - last_update < source_info['update_frequency']:
                    continue

                articles = self._FetchRSSFeed(source_info)
                for article_data in articles:
                    if not self._IsDuplicate(article_data):
                        article = self._CreateNewsArticle(article_data)
                        all_articles.append(article)
                        self.collected_articles[article.article_hash] = article

                self.last_collection_time[source_name] = current_time

            except Exception as e:
                self.logger.error(f"소스 {source_name} 수집 실패: {e}")
                continue

        # 오래된 기사 정리 (메모리 관리)
        self._CleanupOldArticles()

        self.logger.info(f"총 {len(all_articles)}개 새 기사 수집 완료")
        return all_articles

    def _CreateNewsArticle(self, article_data: Dict[str, Any]) -> NewsArticle:
        """딕셔너리 데이터를 NewsArticle 객체로 변환"""
        article_hash = self._CreateArticleHash(
            article_data['title'],
            article_data.get('content', '')
        )

        return NewsArticle(
            title=article_data['title'],
            content=article_data.get('content', ''),
            url=article_data['url'],
            published_time=article_data['published_time'],
            source=article_data['source'],
            language=article_data['language'],
            reliability_score=article_data['reliability_score'],
            article_hash=article_hash
        )

    def _CleanupOldArticles(self):
        """메모리 절약을 위한 오래된 기사 정리"""
        cutoff_time = datetime.now() - timedelta(
            hours=self.collection_config['article_max_age_hours'] * 2
        )

        old_hashes = [
            article_hash for article_hash, article in self.collected_articles.items()
            if article.published_time < cutoff_time
        ]

        for article_hash in old_hashes:
            del self.collected_articles[article_hash]

        if old_hashes:
            self.logger.info(f"{len(old_hashes)}개 오래된 기사 정리 완료")

    def GetCollectionStats(self) -> Dict[str, Any]:
        """수집 통계 정보 반환"""
        return {
            'total_articles': len(self.collected_articles),
            'sources_count': len(self.news_sources['global']) + len(self.news_sources['korean']),
            'last_collection_times': self.last_collection_time,
            'memory_usage_articles': len(self.collected_articles)
        }