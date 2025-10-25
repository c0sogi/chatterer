from typing import Callable, NamedTuple, Self, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class MatchedText(NamedTuple):
    text: str
    start_idx: int
    end_idx: int

    @classmethod
    def from_text(
        cls,
        full_text: str,
        len_func: Callable[[str], int],
        chunk_size: int = 2048,
        token_overlap: int = 0,
        separator: str = "\n",
    ) -> list[Self]:
        """
        토큰 수 제한과 선택적 오버랩을 기준으로 텍스트를 청크로 분할합니다.
        각 청크는 원본 텍스트 내의 위치 정보 (start_idx, end_idx)와 함께 반환됩니다.
        텍스트는 separator 문자열로 분할하며, 토큰 수는 len_func 함수를 통해 계산합니다.

        Args:
            full_text: 분할할 전체 텍스트.
            len_func: 주어진 텍스트의 토큰 수를 반환하는 함수.
            chunk_size: 각 청크의 최대 토큰 수. 기본값은 2048.
            token_overlap: 청크 간 중첩할 토큰 수. 기본값은 0.
            separator: 텍스트를 분할할 구분자 문자열. 기본값은 "\n".

        Returns:
            각 요소가 (chunk_text, start_idx, end_idx)인 튜플의 리스트.
            chunk_text는 whole_text 내에서 whole_text[start_idx:end_idx]와 동일한 부분 문자열입니다.
        """
        text_chunks: list[Self] = []
        sep_token_count: int = len_func(separator)
        sep_len = len(separator)

        # 먼저, separator를 기준으로 원본 텍스트를 분할하되 각 조각의 시작/종료 인덱스를 기록합니다.
        piece_infos: list[Self] = []  # 각 튜플: (piece_text, start_index, end_index)
        start_idx = 0
        while True:
            idx = full_text.find(separator, start_idx)
            if idx == -1:
                # 마지막 조각: separator가 더 이상 없으므로 전체 남은 부분을 추가합니다.
                piece_infos.append(
                    cls(
                        text=full_text[start_idx:],
                        start_idx=start_idx,
                        end_idx=len(full_text),
                    )
                )
                break
            else:
                piece_infos.append(
                    cls(
                        text=full_text[start_idx:idx],
                        start_idx=start_idx,
                        end_idx=idx,
                    )
                )
                start_idx = idx + sep_len

        current_chunk: list[Self] = []
        current_token_count: int = 0
        i = 0
        while i < len(piece_infos):
            piece_info = piece_infos[i]
            piece = piece_info.text
            piece_start = piece_info.start_idx
            piece_end = piece_info.end_idx
            # 원래 코드는 각 조각에 separator의 토큰 수도 포함합니다.
            piece_token_count: int = len_func(piece) + sep_token_count

            # 현재 청크에 추가하면 chunk_size를 초과하는 경우
            if current_token_count + piece_token_count > chunk_size:
                # 단일 조각이 chunk_size보다 큰 경우엔 어쩔 수 없이 추가합니다.
                if not current_chunk:
                    current_chunk.append(
                        cls(
                            text=piece,
                            start_idx=piece_start,
                            end_idx=piece_end,
                        )
                    )
                    current_token_count += piece_token_count
                    i += 1
                # 현재 청크 완성 → 청크에 추가
                chunk_start = current_chunk[0].start_idx
                # current_chunk에 담긴 조각들은 원본 텍스트상 연속되어 있으므로,
                # 청크의 종료 인덱스는 마지막 조각의 end_index가 됩니다.
                chunk_end = current_chunk[-1].end_idx
                # 원본 텍스트의 해당 구간을 그대로 추출하면 separator가 포함됩니다.
                chunk_text = full_text[chunk_start:chunk_end]
                text_chunks.append(
                    cls(
                        text=chunk_text,
                        start_idx=chunk_start,
                        end_idx=chunk_end,
                    )
                )

                # token_overlap이 적용되는 경우: 청크 끝부분 일부를 다음 청크에 오버랩합니다.
                if token_overlap > 0:
                    overlap_chunk: list[Self] = []
                    overlap_count: int = 0
                    # 뒤에서부터 역순으로 오버랩할 조각들을 선택합니다.
                    for j in range(len(current_chunk) - 1, -1, -1):
                        p_text = current_chunk[j].text
                        p_token_count = len_func(p_text) + sep_token_count
                        # 최소 한 조각은 포함하고, 오버랩 토큰 수가 token_overlap 이하라면 계속 추가
                        if overlap_count + p_token_count <= token_overlap or not overlap_chunk:
                            overlap_chunk.insert(0, current_chunk[j])
                            overlap_count += p_token_count
                        else:
                            break
                    current_chunk = overlap_chunk.copy()
                    current_token_count = overlap_count
                else:
                    current_chunk.clear()
                    current_token_count = 0
            else:
                # 청크에 추가 후 다음 조각 진행
                current_chunk.append(cls(text=piece, start_idx=piece_start, end_idx=piece_end))
                current_token_count += piece_token_count
                i += 1

        # 남은 조각이 있다면 마지막 청크로 추가합니다.
        if current_chunk:
            chunk_start = current_chunk[0].start_idx
            chunk_end = current_chunk[-1].end_idx
            chunk_text = full_text[chunk_start:chunk_end]
            text_chunks.append(cls(text=chunk_text, start_idx=chunk_start, end_idx=chunk_end))

        return text_chunks
