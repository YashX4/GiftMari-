// The whole song sequence lives here — add, remove, or reorder tracks by
// editing this array; index.html renders itself from it (see love-notes.js).
//
// lyricsArtist/lyricsTitle are optional overrides for the lyrics lookup
// only, for when the lyrics-provider's best match needs different spelling
// than the artist/title actually credited on Spotify (e.g. transliteration
// differences). Leave them out to just use artist/title as-is.
const TRACKS = [
  {
    order: 1,
    title: "First Day of My Life",
    artist: "Bright Eyes",
    spotifyTrackId: "0eBryM7ePQH3Klt3jz8xZd",
    highlightLyric: { text: "This is the first day of my life", language: "en" },
    feeling: "everything before her feels like a rough draft",
  },
  {
    order: 2,
    title: "Ayonha",
    artist: "Hamid Al Shaeri",
    lyricsArtist: "Hamid El Shaeri",
    spotifyTrackId: "5HCTbcF18u5DcYNwEWWf3n",
    highlightLyric: {
      text: "بشوف عيون قتلاني هزت عقلي ووچداني",
      language: "ar",
      translation: "I see eyes that killed me, they shook my mind and my soul",
    },
    feeling: "the way her eyes are the first thing I notice, every time",
  },
  {
    order: 3,
    title: "Estoy Aquí",
    artist: "Shakira",
    spotifyTrackId: "4M1lEbqPzlEw1JYWB6aE7K",
    highlightLyric: {
      text: "Estoy aquí, queriéndote",
      language: "es",
      translation: "I'm here, loving you",
    },
    feeling: "already sang this with her once — it's imprinted now",
  },
  {
    order: 4,
    title: "My Love",
    displayTitle: "My Love (feat. T.I.)",
    artist: "Justin Timberlake, T.I.",
    lyricsArtist: "Justin Timberlake",
    lyricsTitle: "My Love",
    spotifyTrackId: "4NeOWqHmlrGRuBvsLJC9rL",
    highlightLyric: { text: "All I want you to do is be my love", language: "en" },
    feeling: "smooth and sure, zero doubts",
  },
  {
    order: 5,
    title: "Romeo",
    artist: "PinkPantheress",
    spotifyTrackId: "6POxiQbr5dFg2gU68yh4NK",
    highlightLyric: { text: "You're all I can imagine", language: "en" },
    feeling: "bouncy and a little silly, like her laugh",
  },
  {
    order: 6,
    title: "I'm a Believer",
    artist: "The Monkees",
    spotifyTrackId: "3G7tRC24Uh09Hmp1KZ7LQ2",
    highlightLyric: { text: "Then I saw her face, now I'm a believer", language: "en" },
    feeling: "the corniest, most classic she-walked-in-and-I-was-done-for song",
  },
  {
    order: 7,
    title: "Kokomo",
    artist: "The Beach Boys",
    spotifyTrackId: "5qHYXcVvc9xsFB2uH7GpMN",
    highlightLyric: { text: "Aruba, Jamaica, ooh I wanna take ya", language: "en" },
    feeling: "daydreaming about running off somewhere dumb and tropical with her",
  },
];
