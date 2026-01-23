import os
import tempfile

import pytest

import dspy
from dspy.utils.dummies import DummyLM


@pytest.fixture
def sample_video_file():
    """Create a small fake video file for testing."""
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(b"\x00\x00\x00\x1c\x66\x74\x79\x70\x69\x73\x6f\x6d")
        tmp_file.write(b"\x00" * 100)
        tmp_file_path = tmp_file.name
    yield tmp_file_path
    try:
        os.unlink(tmp_file_path)
    except Exception:
        pass


def count_messages_with_file_pattern(messages):
    """Count the number of file content blocks in messages."""
    pattern = {"type": "file", "file": lambda x: isinstance(x, dict)}

    def check_pattern(obj, pattern):
        if isinstance(pattern, dict):
            if not isinstance(obj, dict):
                return False
            return all(k in obj and check_pattern(obj[k], v) for k, v in pattern.items())
        if callable(pattern):
            return pattern(obj)
        return obj == pattern

    def count_patterns(obj, pattern):
        count = 0
        if check_pattern(obj, pattern):
            count += 1
        if isinstance(obj, dict):
            count += sum(count_patterns(v, pattern) for v in obj.values())
        if isinstance(obj, list | tuple):
            count += sum(count_patterns(v, pattern) for v in obj)
        return count

    return count_patterns(messages, pattern)


def setup_predictor(signature, expected_output):
    lm = DummyLM([expected_output])
    dspy.settings.configure(lm=lm)
    return dspy.Predict(signature), lm


# ============================================================================
# Basic signature tests
# ============================================================================


class TestVideoInSignature:
    def test_video_basic_signature(self, sample_video_file):
        """Test video in a basic signature."""
        signature = "video: dspy.Video -> description: str"
        expected = {"description": "A video showing something"}
        predictor, lm = setup_predictor(signature, expected)

        video = dspy.Video.from_path(sample_video_file)
        result = predictor(video=video)

        assert result.description == "A video showing something"
        assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 1

    def test_video_with_text_input(self, sample_video_file):
        """Test video combined with text input."""

        class VideoQA(dspy.Signature):
            video: dspy.Video = dspy.InputField()
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        expected = {"answer": "The video shows a person walking"}
        predictor, lm = setup_predictor(VideoQA, expected)

        video = dspy.Video.from_path(sample_video_file)
        result = predictor(video=video, question="What is happening in this video?")

        assert result.answer == "The video shows a person walking"
        assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 1

    def test_video_from_url_in_signature(self):
        """Test video from URL in signature."""

        class VideoSummary(dspy.Signature):
            video: dspy.Video = dspy.InputField()
            summary: str = dspy.OutputField()

        expected = {"summary": "Summary of video content"}
        predictor, lm = setup_predictor(VideoSummary, expected)

        video = dspy.Video.from_url("https://example.com/video.mp4")
        result = predictor(video=video)

        assert result.summary == "Summary of video content"
        assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 1

    def test_video_from_youtube_in_signature(self):
        """Test YouTube video in signature."""

        class YouTubeAnalysis(dspy.Signature):
            video: dspy.Video = dspy.InputField()
            analysis: str = dspy.OutputField()

        expected = {"analysis": "This YouTube video discusses..."}
        predictor, lm = setup_predictor(YouTubeAnalysis, expected)

        video = dspy.Video.from_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        result = predictor(video=video)

        assert result.analysis == "This YouTube video discusses..."
        assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 1

    def test_video_from_file_id_in_signature(self):
        """Test video from pre-uploaded file_id in signature."""

        class VideoCaption(dspy.Signature):
            video: dspy.Video = dspy.InputField()
            caption: str = dspy.OutputField()

        expected = {"caption": "A beautiful sunset"}
        predictor, lm = setup_predictor(VideoCaption, expected)

        video = dspy.Video.from_file_id("files/abc123", mime_type="video/mp4")
        result = predictor(video=video)

        assert result.caption == "A beautiful sunset"
        assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 1


# ============================================================================
# List of videos tests
# ============================================================================


class TestVideoListInSignature:
    def test_video_list_basic(self, sample_video_file):
        """Test list of videos in signature."""

        class MultiVideoSignature(dspy.Signature):
            videos: list[dspy.Video] = dspy.InputField()
            comparison: str = dspy.OutputField()

        expected = {"comparison": "The videos show different scenes"}
        predictor, lm = setup_predictor(MultiVideoSignature, expected)

        videos = [
            dspy.Video.from_path(sample_video_file),
            dspy.Video.from_url("https://example.com/video2.mp4"),
        ]
        result = predictor(videos=videos)

        assert result.comparison == "The videos show different scenes"
        assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 2

    def test_mixed_video_sources_list(self, sample_video_file):
        """Test list with videos from different sources."""

        class MixedVideoSignature(dspy.Signature):
            videos: list[dspy.Video] = dspy.InputField()
            summary: str = dspy.OutputField()

        expected = {"summary": "Combined video analysis"}
        predictor, lm = setup_predictor(MixedVideoSignature, expected)

        videos = [
            dspy.Video.from_path(sample_video_file),
            dspy.Video.from_youtube("https://www.youtube.com/watch?v=test"),
            dspy.Video.from_file_id("files/xyz789"),
        ]
        result = predictor(videos=videos)

        assert result.summary == "Combined video analysis"
        assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 3


# ============================================================================
# Optional video tests
# ============================================================================


class TestOptionalVideoInSignature:
    def test_optional_video_with_value(self, sample_video_file):
        """Test optional video field with a value."""

        class OptionalVideoSignature(dspy.Signature):
            video: dspy.Video | None = dspy.InputField()
            output: str = dspy.OutputField()

        expected = {"output": "Video processed"}
        predictor, lm = setup_predictor(OptionalVideoSignature, expected)

        video = dspy.Video.from_path(sample_video_file)
        result = predictor(video=video)

        assert result.output == "Video processed"
        assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 1

    def test_optional_video_with_none(self):
        """Test optional video field with None."""

        class OptionalVideoSignature(dspy.Signature):
            video: dspy.Video | None = dspy.InputField()
            output: str = dspy.OutputField()

        expected = {"output": "No video provided"}
        predictor, lm = setup_predictor(OptionalVideoSignature, expected)

        result = predictor(video=None)

        assert result.output == "No video provided"
        assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 0


# ============================================================================
# Video with other media types tests
# ============================================================================


class TestVideoWithOtherMedia:
    def test_video_with_image(self, sample_video_file):
        """Test video combined with image in same signature."""

        class MultiMediaSignature(dspy.Signature):
            video: dspy.Video = dspy.InputField()
            image: dspy.Image = dspy.InputField()
            analysis: str = dspy.OutputField()

        expected = {"analysis": "Combined media analysis"}
        predictor, lm = setup_predictor(MultiMediaSignature, expected)

        video = dspy.Video.from_path(sample_video_file)
        # Use a simple image URL
        image = dspy.Image("https://example.com/image.jpg")

        result = predictor(video=video, image=image)

        assert result.analysis == "Combined media analysis"
        # Check both video (file type) and image (image_url type) are present
        messages = lm.history[-1]["messages"]
        assert count_messages_with_file_pattern(messages) == 1  # Video

        # Count image_url patterns
        def count_image_patterns(obj):
            if isinstance(obj, dict):
                if obj.get("type") == "image_url":
                    return 1
                return sum(count_image_patterns(v) for v in obj.values())
            if isinstance(obj, list | tuple):
                return sum(count_image_patterns(v) for v in obj)
            return 0

        assert count_image_patterns(messages) == 1  # Image


# ============================================================================
# Save/load tests
# ============================================================================


class TestVideoSaveLoad:
    def test_save_load_video_predictor(self, sample_video_file):
        """Test saving and loading a predictor with video examples."""
        signature = "video: dspy.Video -> description: str"
        video = dspy.Video.from_path(sample_video_file)
        examples = [dspy.Example(video=video, description="Test description")]

        predictor, lm = setup_predictor(signature, {"description": "A description"})
        optimizer = dspy.teleprompt.LabeledFewShot(k=1)
        compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

        with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
            compiled_predictor.save(temp_file.name)
            loaded_predictor = dspy.Predict(signature)
            loaded_predictor.load(temp_file.name)

        # Run prediction with loaded predictor
        loaded_predictor(video=dspy.Video.from_file_id("files/test"))

        # Should have 2 videos: one from few-shot example, one from input
        assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 2


# ============================================================================
# String-based signature tests
# ============================================================================


class TestVideoStringSignature:
    def test_string_signature_with_video_type(self, sample_video_file):
        """Test video type in string-based signature."""
        signature = "video: dspy.Video, prompt: str -> response: str"
        expected = {"response": "Here is the analysis"}
        predictor, lm = setup_predictor(signature, expected)

        video = dspy.Video.from_path(sample_video_file)
        result = predictor(video=video, prompt="Analyze this video")

        assert result.response == "Here is the analysis"
        assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 1


# ============================================================================
# Chain of Thought with video tests
# ============================================================================


class TestVideoChainOfThought:
    def test_cot_with_video(self, sample_video_file):
        """Test ChainOfThought module with video input."""

        class VideoAnalysis(dspy.Signature):
            """Analyze a video and provide detailed description."""

            video: dspy.Video = dspy.InputField(desc="The video to analyze")
            description: str = dspy.OutputField(desc="Detailed description of the video")

        expected = {
            "reasoning": "Looking at the video frames...",
            "description": "The video shows a scenic landscape",
        }
        lm = DummyLM([expected])
        dspy.settings.configure(lm=lm)

        cot = dspy.ChainOfThought(VideoAnalysis)
        video = dspy.Video.from_path(sample_video_file)
        result = cot(video=video)

        assert result.description == "The video shows a scenic landscape"
        assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 1
